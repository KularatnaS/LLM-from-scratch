import unittest

import numpy as np
import torch
import torch.nn as nn

from model.model import InputEmbeddings, PositionalEncoding, LayerNormalization, FeedForwardBlock, MultiHeadAttentionBlock, \
    ResidualConnection, EncoderBlock, Encoder, ProjectionLayer


import logging
LOGGER = logging.getLogger(__name__)


class Test_3d_transformer_model(unittest.TestCase):

    def test_input_embeddings(self):
        # GIVEN
        d_model = 512
        vocab_size = 100
        input_embeddings = InputEmbeddings(d_model, vocab_size)

        # WHEN
        x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        output = input_embeddings(x)

        # THEN
        self.assertEqual(output.shape, (2, 3, d_model))

    def test_positional_encoding(self):
        # GIVEN
        d_model = 10
        seq_len = 5
        batch_size = 2
        dropout = 0.0
        positional_encoding = PositionalEncoding(d_model, seq_len, dropout)

        # WHEN
        x = torch.ones(batch_size, seq_len, d_model)
        output = positional_encoding(x)

        # THEN
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))
        self.assertTrue(np.allclose(output[0], output[1]))

    def test_layer_normalization(self):
        # GIVEN
        d_model = 10
        seq_len = 5
        batch_size = 2
        eps = 10**-6
        layer_normalization = LayerNormalization(eps)

        # WHEN
        x = torch.rand(batch_size, seq_len, d_model)
        output = layer_normalization(x)

        # THEN
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))

    def test_feed_forward_block(self):
        # GIVEN
        d_model = 10
        d_ff = 20
        dropout = 0.1
        seq_len = 5
        batch_size = 2
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)

        # WHEN
        x = torch.rand(batch_size, seq_len, d_model)
        output = feed_forward_block(x)

        # THEN
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))

    def test_multi_head_attention_block(self):
        # GIVEN
        d_model = 256
        dropout = 0.1
        h = 8
        seq_len = 5
        batch_size = 2
        multi_head_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        x = torch.rand(batch_size, seq_len, d_model)

        # WHEN / THEN
        output = multi_head_attention_block(x, x, x, None)
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))

        # WHEN / THEN
        mask = torch.zeros(batch_size, h, seq_len, seq_len)
        output = multi_head_attention_block(x, x, x, mask)
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))

    def test_residual_connection(self):
        # GIVEN
        d_model = 256
        d_ff = 512
        h = 8
        seq_len = 5
        batch_size = 2
        dropout = 0.1
        residual_connection = ResidualConnection(dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        x = torch.rand(batch_size, seq_len, d_model)

        # WHEN / THEN
        output = residual_connection(x, feed_forward_block)
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))

    def test_encoder_block(self):
        # GIVEN
        d_model = 256
        d_ff = 512
        h = 8
        seq_len = 5
        batch_size = 2
        dropout = 0.1
        multi_head_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(multi_head_attention_block, feed_forward_block, dropout)
        x = torch.rand(batch_size, seq_len, d_model)

        # WHEN / THEN
        output = encoder_block(x, None)
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))

        # WHEN / THEN
        mask = torch.zeros(batch_size, h, seq_len, seq_len)
        output = encoder_block(x, mask)
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))

    def test_encoder(self):
        # GIVEN
        d_model = 256
        d_ff = 512
        h = 8
        seq_len = 5
        batch_size = 2
        dropout = 0.1
        multi_head_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(multi_head_attention_block, feed_forward_block, dropout)
        encoder = Encoder(nn.ModuleList([encoder_block, encoder_block, encoder_block, encoder_block]))
        x = torch.rand(batch_size, seq_len, d_model)

        # WHEN / THEN
        output = encoder(x, None)
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))

        # WHEN / THEN
        mask = torch.zeros(batch_size, h, seq_len, seq_len)
        output = encoder(x, mask)
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))

    def test_projection_layer(self):
        # GIVEN
        d_model = 4
        vocab_size = 10
        batch_size = 1
        seq_len = 5
        projection_layer = ProjectionLayer(d_model, vocab_size)
        x = torch.rand(batch_size, seq_len, d_model)

        # WHEN / THEN
        output = projection_layer(x)
        self.assertEqual(output.shape, (batch_size, seq_len, vocab_size))


if __name__ == '__main__':
    unittest.main()
