from typing import List
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode
# nltk.download()
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt_tab')
# download from "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip"
# unzip and create a folder named nltk_data inside virtual env like \.virtualenvs\FinSentimentAnalyzer-enDW9jvo 
# place unzipped folder inside nltk_data



class PreprocessPipe:
    def __init__(self) -> None:
        self.__func_order__=[]
        self.__stop_words__ = set(stopwords.words('english'))
        self.__word_tokenize__=word_tokenize
        self.__stemmer__ = PorterStemmer()
        self.__lemmatizer__ = WordNetLemmatizer()
    def get_order(self):
        
        funcs=""
        for i,func in enumerate(self.__func_order__):
            funcs+= f'{i+1} . {func.__name__} \n'
        return funcs

    def run(self,texts:List[str]):
        for func in self.__func_order__:
            texts=func(texts)
        return texts
    
    def add_steps(self,func_names : List[str]):
        """
        available preprocessing steps:

        lowercase_converter

        remove_extra_space

        remove_numbers
        
        remove_special_character

        remove_bracket_content

        remove_stopwords

        do_stemming

        do_lemmatization

        fix_encoding

        """
        self.__func_order__.extend(func_names)

    def lowercase_converter(self,texts):
        return [text.lower() for text in texts]

    def remove_extra_space(self,texts:List[str]):

        return [re.sub(r'\s+', ' ', text).strip() for text in texts]

    def remove_numbers(self,texts:List[str]):
         return [re.sub(r'\d+', '', text) for text in texts]
         
    def remove_special_character(self,texts:List[str]):
        return [re.sub(r'[^a-zA-Z0-9\s]', '', text)  for text in texts]
    

    def remove_bracket_content(self,texts:List[str]):
        result=[]
        for text in texts:
            text=re.sub(r'\[.*?\]', '', text)
            result.append(re.sub(r'\(.*?\)', '', text))
        return result
    
    def remove_stopwords(self,texts:List[str]):
        """
        
        """
        result=[]
        for text in texts: 
            words=self.__word_tokenize__(text)
            filtered_words = [word for word in words if word.lower() not in self.__stop_words__]
            result.append(' '.join(filtered_words))
        return result


    def do_stemming(self,texts:List[str]):
        result=[]
        for text in texts: 
            words=self.__word_tokenize__(text)
            filtered_words = [self.__stemmer__.stem(word) for word in words ]
            result.append(' '.join(filtered_words))
        return result
    
    def do_lemmatization(self,texts:List[str]):
            result=[]
            for text in texts: 
                words=self.__word_tokenize__(text)
                filtered_words = [self.__lemmatizer__.lemmatize(word) for word in words ]
                result.append(' '.join(filtered_words))
            return result

    def fix_encoding(self,texts:List[str]):
        
        return [unidecode(text) for text in texts]

# input=["In Q1 of 2010 , Bank of +Ã land 's net interest income increased from EUR 9.1 mn to EUR 9.7 mn ."]
# pipe=PreprocessPipe()
# pipe.add_steps([pipe.fix_encoding ,
#                 pipe.lowercase_converter,pipe.remove_numbers,pipe.remove_special_character, 
#                 pipe.remove_bracket_content, pipe.do_lemmatization ,pipe.remove_extra_space ])
# print(input)
# print(pipe.get_order())

# print(pipe.run(input))
# # print(pipe.__stop_words__)
