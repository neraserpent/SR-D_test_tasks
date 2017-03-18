#Python v3.5
import numpy
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.optimizers import RMSprop
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import sys

#original_text = open('Tolstoy_merged.txt').read().lower()
# Объём полного текста слишком велик для моих вычислительных ресурсов
original_text = open('Tolstoy_book1.txt').read().lower()
chars = sorted(list(set(original_text)))
index_to_char = dict((i, c) for i, c in enumerate(chars))
char_to_index = dict((c, i) for i, c in enumerate(chars))

text_len = len(original_text)
print("Length of file: ", len(original_text))
print("Length of char vocabulary: ", len(chars))

def text_to_vectors(text, seq_len = 50, fraction = 1):
        """
        Разделение текста на блоки длины seq_len с посимвольным шагом
        и векторизация

        text - исходный текст в полном объёме
        seq_len - длина части текста, подающейся на входной слой НС
        fraction - делитель размера обучающей выборки
        """
        sequences = []
        targets = []
        for i in range(0, int((len(text) - seq_len)/fraction)): 
                sequence = text[i:i + seq_len]
                target = text[i + seq_len]
                sequences.append([char_to_index[char] for char in sequence])
                targets.append(char_to_index[target])

        # Векторизация для входного слоя
        X = numpy.reshape(sequences, (len(sequences), seq_len, 1))
        # Нормализация для сигмоидной активационной функции
        X = X / float(len(chars)) 
        y = np_utils.to_categorical(targets) # One-hot кодировка 

        return X, y

def generate_chunk(model, text, seq_len = 50, chunk_len = 200):
        """
        Функция генерации текста на основе обученной модели

        model - обученная модель
        text - исходный текст в объёме не менее seq_len
        seq_len - длина части текста, подающейся на входной слой НС
        chunk_len - длина генерируемого текста
        """
        start_index = numpy.random.randint(0, len(text) - seq_len -1)
        start_sequence = text[start_index:start_index + seq_len]
        sequence = [char_to_index[char] for char in start_sequence]
        #sequence = numpy.reshape(sequence, (1, seq_len, 1))
        result_string = []
        for i in range(100):
                x = numpy.reshape(sequence, (1, len(sequence), 1))
                x = x / float(len(chars))
                prediction = model.predict(x, verbose=0)
                index = numpy.argmax(prediction)
                result = index_to_char[index]
                result_string.append(result)
                sequence.append(index)
                sequence = sequence[1:len(sequence)]
        return ''.join(result_string)

X, y = text_to_vectors(original_text, fraction = 1)
# Определение топологии сети, компиляция и старт обучения
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))
# Cохранение модели и весов после каждой эпохи
checkpoint = ModelCheckpoint("check_{epoch:02d}_{loss:.4f}.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='min')
model.fit(X, y, epochs=40, batch_size=128, callbacks=[checkpoint])

print(generate_chunk(model, original_text))


