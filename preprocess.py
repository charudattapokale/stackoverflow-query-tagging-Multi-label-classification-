# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing import text
from sklearn.preprocessing import MultiLabelBinarizer


raw_data = pd.read_csv("./2/stack_data_final.csv")

raw_data = shuffle(raw_data, random_state = 420)

#%%

tags_split = [tags.split() for tags in raw_data['tags'] ]

print(tags_split)
#%%

each_tag_raw_data = [each for sublist in tags_split for each in sublist ]

#%%

each_tag_count_rawdata = pd.Series(each_tag_raw_data).value_counts().reset_index().rename(columns = {'index':'tags',0:'counts' } )


#%%
              
keeplist = ['android','javascript','java','c++','c#','asp.net','php','jquery','.net']
              
def keep_tags(lis,keeplist = keeplist):
    
    splitt = lis.split()
    
    finall = [fin if fin in keeplist else 'other' for fin in splitt]
    
    finall = list(set(finall))
    
    return finall
   
#%%
data_keeplist = raw_data.copy()
data_keeplist['label'] = data_keeplist['tags'].apply(keep_tags)

#%%

each_tag_keeplist = [each for sublist in data_keeplist['label'] for each in sublist ]

each_tag_count_keeplist = pd.Series(each_tag_keeplist).value_counts().reset_index().rename(columns = {'index':'tags',0:'counts' } )

#%%
only_other_label =  [sublist for sublist in data_keeplist['label'] if set(sublist) == {'other'} ]
#%%


data_keeplist['only_other'] = data_keeplist['label'].apply(lambda x: True if x == ['other'] else False)

print(data_keeplist.count())


data_keeplist = data_keeplist.drop(data_keeplist[data_keeplist['only_other'] == True].index)

print(data_keeplist.count())
#%%
print(data_keeplist['question'].head(20))

data_keeplist['question'] = data_keeplist['question'].replace( keeplist ,'XXX',regex = True)
    
print(data_keeplist['question'].head(20))
    
#%%
trail = data_keeplist.copy()
cnt = 0

# =============================================================================
# for row in trail.iterrows():
#     if set(row[1]['label']) == {'other'}:
#         trail = trail.drop(row[0])
#         cnt += 1
#         print(cnt)
#         if cnt%10 == 0:
#             print("row {0} deleted cnt is {1}".format(row[0],cnt) )
# =============================================================================
        
#%%



tag_encoder = MultiLabelBinarizer()
tags_encoded = tag_encoder.fit_transform(data_keeplist['label'])
num_tags = len(tags_encoded[0])
print(data_keeplist['question'].values[0])
print(tag_encoder.classes_)
print(tags_encoded[0])

 #%%

target_train = tags_encoded[0:499000]
target_test = tags_encoded[499000:]

 #%%

from tensorflow.keras.preprocessing import text

VOCAB_SIZE = 100

def get_token(text_list,num_words = VOCAB_SIZE):
    tokenizer = text.Tokenizer(num_words)
    tokenizer.fit_on_texts(text_list)
    return tokenizer
    

def txt2mat(tokenizer,text_list):
    text_matrix = tokenizer.texts_to_matrix(text_list)
    return text_matrix
 
 #%%

train_size = 499000
 
train_qs = data_keeplist['question'].values[:train_size]
test_qs = data_keeplist['question'].values[train_size:]

processor = get_token(train_qs)

body_train = txt2mat(processor, train_qs)
body_test = txt2mat(processor, test_qs)

 #%%
import pickle

with open('./processor_state.pkl', 'wb') as f:
  pickle.dump(processor, f)
 

 #%%
import tensorflow as tf 


 
def create_model(vocab_size, num_tags):
  
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Dense(50, input_shape=(VOCAB_SIZE,), activation='relu'))
  model.add(tf.keras.layers.Dense(25, activation='relu'))
  model.add(tf.keras.layers.Dense(num_tags, activation='sigmoid'))

  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model 
#%%

model = create_model(VOCAB_SIZE, num_tags)
model.summary()

# Train and evaluate the model
model.fit(body_train, target_train, epochs=15, batch_size=64, validation_split=0.1, verbose=1)
print('Eval loss/accuracy:{}'.format(
  model.evaluate(body_test, target_test, batch_size=64)))

# Export the model to a file
model.save('keras_saved_model_version1.h5')

#%%
