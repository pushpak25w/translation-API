import os
import pandas as pd
import numpy as np
import string
from string import digits
import re
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model,load_model
import pickle
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

class LSTM_model():
    def __init__(self,langDir,transDir,modelDir,max_length_src,max_length_tar,latent_dim):
        self.spchrs=set(string.punctuation)
        self.lowerCase=lambda x:x.lower()
        self.quotes=lambda x:re.sub("'",'',x)
        self.specialCh=lambda x:''.join(ch for ch in x if ch not in self.spchrs)
        self.rmDigits=lambda x:x.translate(str.maketrans('','',digits))
        self.spaces=lambda x:x.strip()
        self.unwanted=lambda x:re.sub(" +"," ",x)
        self.startEnd=lambda x:'START_ '+x+' _END'
        all_lang_words=pickle.load(open(langDir, "rb"))
        all_trans_words=pickle.load(open(transDir, "rb"))
        self.max_length_src=max_length_src
        input_words = sorted(list(all_lang_words))
        target_words = sorted(list(all_trans_words))
        num_encoder_tokens = len(all_lang_words)
        num_decoder_tokens = len(all_trans_words)
        num_decoder_tokens+=1
        self.input_token_index = dict([(word, i+1) for i, word in enumerate(input_words)])
        self.target_token_index = dict([(word, i+1) for i, word in enumerate(target_words)])
        reverse_input_char_index = dict((i, word) for word, i in self.input_token_index.items())
        self.reverse_target_char_index = dict((i, word) for word, i in self.target_token_index.items())
        encoder_inputs = Input(shape=(None,))
        enc_emb =  Embedding(num_encoder_tokens, latent_dim, mask_zero = True)(encoder_inputs)
        encoder_lstm = LSTM(latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
        encoder_states = [state_h, state_c]
        decoder_inputs = Input(shape=(None,))
        dec_emb_layer = Embedding(num_decoder_tokens, latent_dim, mask_zero = True)
        dec_emb = dec_emb_layer(decoder_inputs)
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(dec_emb,initial_state=encoder_states)
        decoder_dense = Dense(num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        model.load_weights(modelDir)
        self.encoder_model = Model(encoder_inputs, encoder_states)
        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        dec_emb2= dec_emb_layer(decoder_inputs)
        decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
        decoder_states2 = [state_h2, state_c2]
        decoder_outputs2 = decoder_dense(decoder_outputs2)
        self.decoder_model = Model([decoder_inputs] + decoder_states_inputs,[decoder_outputs2] + decoder_states2)

    def shape_data(self,sentence):
        sentence=self.quotes(sentence)
        sentence=self.specialCh(sentence)
        sentence=self.rmDigits(sentence)
        sentence=self.spaces(sentence)
        sentence=self.unwanted(sentence)
        sentence=self.lowerCase(sentence)
        return sentence

    def pad_data(self,sentence):
        encoder_input_data = np.zeros((1, self.max_length_src),dtype='float32')
        for t, word in enumerate(sentence.split()):
            encoder_input_data[0, t] = self.input_token_index[word]
        return encoder_input_data

    def decode_sequence(self,input_seq):
        states_value = self.encoder_model.predict(input_seq)
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = self.target_token_index['START_']
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.reverse_target_char_index[sampled_token_index]
            decoded_sentence += ' '+sampled_char
            if (sampled_char == '_END' or len(decoded_sentence) > 50):
                stop_condition = True
            target_seq = np.zeros((1,1))
            target_seq[0, 0] = sampled_token_index
            states_value = [h, c]
        return decoded_sentence
