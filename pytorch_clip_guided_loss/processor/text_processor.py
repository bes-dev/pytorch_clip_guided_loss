"""
Copyright 2021 by Sergei Belousov
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import typing
import torch
import youtokentome as yttm
from transformers import CLIPTokenizer
from googletrans import Translator
from .utils import tokens_to_tensor


class GoogleTranslate:
    """ Implementation of the API to translate.google.com
    Arguments:
        target_lang (str): target language.
    """
    def __init__(self, target_lang: str = 'ru'):
        self.target_lang = target_lang
        self.translator = Translator()

    def __call__(self, text: str):
        """ Translate input text to target language if needs.
        Arguments:
            text (str): input text.
        Returns:
            translation (str): output text in target language.
        """
        try:
            source_lang = self.translator.detect(text).lang
            if source_lang != self.target_lang:
                print(f"translate {source_lang}->{self.target_lang}")
                translation = self.translator.translate(text, dest=self.target_lang).text
            else:
                translation = text
        except:
            translation = text
        return translation


class TextProcessor:
    """ Base class for text processor.
    Arguments:
        target_lang (str): target language.
    """
    def __init__(self, target_lang: typing.Optional[str] = None):
        self.translator = GoogleTranslate(target_lang) if target_lang is not None else None

    def __len__(self) -> int:
        """ Return vocabulary size.
        Returns:
            length (int): vocabulary size.
        """
        return self.vocab_size()

    def vocab_size(self) -> int:
        """ Return vocabulary size.
        Returns:
            length (int): vocabulary size.
        """
        raise NotImplemented

    def encode(
            self,
            text: str,
            *args, **kwargs
    ) -> typing.Dict[str, torch.Tensor]:
        """ Encode input text.
        Arguments:
            text (str): input text.
            *args, **kwargs: auxiliary arguments.
        Returns:
            encoded_text (dict<str:torch.Tensor>): encoded text.
        """
        raise NotImplemented

    def decode(self, tokens: torch.Tensor) -> str:
        """ Decode input tokens.
        Arguments:
            tokens (torch.Tensor): input tokens.
        Returns:
            text (str): decoded text.
        """
        raise NotImplemented

    def translate(self, text: str) -> str:
        """ Translate text if needed.
        Arguments:
            text (str): input text.
        Returns:
            text (str): output text.
        """
        if self.translator is not None:
            text = self.translator(text)
        return text


class YTTMTokenizerTextProcessor(TextProcessor):
    """ Implementation of the YTTM tokenizer.
    Arguments:
        model_path (str): path to pretrained model.
        target_length (int): target length of the encoded tokens.
        target_lang (str): target language.
    """
    eos_id = 3
    bos_id = 2
    unk_id = 1
    pad_id = 0

    def __init__(self, model_path: str, target_length=None, target_lang=None):
        super().__init__(target_lang)
        self.target_length = target_length
        self.tokenizer = yttm.BPE(model=model_path)

    def vocab_size(self) -> int:
        """ Return vocabulary size.
        Returns:
            length (int): vocabulary size.
        """
        return self.tokenizer.vocab_size()

    def encode(
            self,
            text: str,
            to_lowercase: bool = True,
            dropout_prob: float = 0.0,
            return_mask: bool = False
    ) -> typing.Dict[str, torch.Tensor]:
        """ Encode input text.
        Arguments:
            text (str): input text.
            to_lowercase (bool): edit text to lowercase.
            dropout_prob (float): dropout prob.
            return_mask (bool): return attention mask
        Returns:
            encoded_text (dict<str:torch.Tensor>): encoded text.
        """
        text = self.translate(text)
        if to_lowercase:
            text = text.lower()
        tokens = self.tokenizer.encode(
            [text.strip()],
            output_type=yttm.OutputType.ID,
            dropout_prob=dropout_prob
        )[0]
        tokens = [self.bos_id] + tokens + [self.eos_id]
        if self.target_length is None:
            target_length = len(tokens)
        else:
            target_length = self.target_length
        output = {}
        output["input_ids"] = tokens_to_tensor(tokens, target_length, self.pad_id)
        if return_mask:
            output["attention_mask"] = (output["input_ids"] != self.pad_id)
        return output

    def decode(self, tokens: torch.Tensor) -> str:
        """ Decode input tokens.
        Arguments:
            tokens (torch.Tensor): input tokens.
        Returns:
            text (str): decoded text.
        """
        return self.tokenizer.decode(
            tokens.cpu().numpy().tolist(),
            ignore_ids=[self.eos_id, self.bos_id, self.unk_id, self.pad_id]
        )[0]


class HFCLIPTextProcessor(TextProcessor):
    """ Implementation of the Huggingface CLIP tokenizer.
    Arguments:
        tokenizer (CLIPTokenizer): pre-trained CLIP tokenizer.
        target_lang (str): target language.
    """
    def __init__(self, tokenizer: CLIPTokenizer, target_lang: typing.Optional[str] = None):
        super().__init__(target_lang)
        self.tokenizer = tokenizer

    def vocab_size(self) -> int:
        """ Return vocabulary size.
        Returns:
            length (int): vocabulary size.
        """
        return self.tokenizer.vocab_size()

    def encode(
            self,
            text: str,
            return_tensors: str = "pt",
            return_mask: bool = True,
            **kwargs
    ) -> typing.Dict[str, torch.Tensor]:
        """ Encode input text.
        Arguments:
            text (str): input text.
            return_tensor (str): return tokens as torch.Tensor.
            return_mask (bool): return attention mask
            **kwargs: auxiliary arguments.
        Returns:
            encoded_text (dict<str:torch.Tensor>): encoded text.
        """
        text = self.translate(text)
        output = self.tokenizer(text, return_tensors=return_tensors, **kwargs)
        if not return_mask:
             del output["attention_mask"]
        return output

    def decode(
            self,
            tokens: torch.Tensor,
            skip_special_tokens: bool = True,
            clean_up_tokenization_spaces: bool = True
    ) -> str:
        """ Decode input tokens.
        Arguments:
            tokens (torch.Tensor): input tokens.
            skip_special_tokens (bool): clean special tokens.
            clean_up_tokenization_spaces (bool): clean special tokens.
        Returns:
            text (str): decoded text.
        """
        if tokens.ndim == 1:
            return self.tokenizer.decode(
                tokens,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces
            )
        else:
            return self.tokenizer.decode_batch(
                tokens,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces
            )
