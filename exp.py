from preprocess import PreprocessPipe
from datasets import load_dataset
from model import DistilBert
from preprocess import PreprocessPipe
class Experiment:
    def __init__(self,name:str, preprocess_pipeline:PreprocessPipe,model:DistilBert):
        self.exp_name=name
        self.train_data_path=f"./data/final_train_dataset.csv"
        self.primary_test_data_path=f"./data/final_primary_dataset.csv"
        self.secondary_test_data_path=f"./data/final_secondary_dataset.csv"
        self.input_column='sentence'
        self.output_column='sentiment'
        self.correct_output_column='labels'
        self.preprocesspipe =preprocess_pipeline
        self.model=model
        self.dataset=None
    def __transform__(self,row):
        processed_text=self.preprocesspipe.run(row["sentence"])
        return self.model.tokenize_data(processed_text)

    def __transform_labels__(self,sample):

        sample = sample['labels']
        num = 0
        if sample == -1: #Negative
            num = 0
        elif sample == 0: #Neutral
            num = 1
        elif sample == 1: #Positive
            num = 2

        return {'labels': num}

    def load_transform_dataset(self):
        dataset=load_dataset(
                'csv',
                data_files={
                    'train': self.train_data_path,
                    'primary_test': self.primary_test_data_path,
                    'secondary_test': self.secondary_test_data_path
                },
                
     trust_remote_code=True
            )
        dataset = dataset.rename_column(self.output_column, self.correct_output_column) #trainer object will consider labels as target
        dataset = dataset.map(self.__transform_labels__)# for classification we need +ve int labels
        dataset = dataset.map(self.__transform__, batched=True) 
        self.train_dataset = dataset['train'].shuffle(seed=10)
        self.primary_test_dataset = dataset['primary_test'].shuffle(seed=10)
        self.secondary_test_dataset = dataset['secondary_test'].shuffle(seed=10)


        self.train_dataset=self.train_dataset.select(range(100))
    

    def finetune(self):
        self.model.finetune(self.train_dataset,self.primary_test_dataset)
    
    def evaluate_test_data(self):
         primary_result=self.model.evaluate_primary_test()
         secondary_result=self.model.evaluate_secondary_test(sec_data=self.secondary_test_dataset)
         return primary_result,secondary_result


    
    def save_model(self):
        self.model.save_model_tokenizer(path=self.exp_name+"/finetune")
        
    
 
    # def add_data_to_ml_flow(self):
    #         mlflow.set_experiment(self.exp_name)
    #         mlflow.start_run():
    #             # Log parameters
    #             mlflow.log_param("preprocess",self.preprocesspipe.get_order() )
    #             mlflow.log_param("model training config", self.model.training_params)
                    