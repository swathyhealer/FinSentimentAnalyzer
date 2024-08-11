from transformers import AutoTokenizer , AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
import numpy as np
from datasets import load_metric


class DistilBert:
    def __init__(self,exp_name,training_params={"num_train_epochs":1, 
                                  "load_best_model_at_end":True,
                                  "evaluation_strategy":'epoch', 
                                  "save_strategy":"epoch",
                                  "learning_rate":2e-5,}) -> None:
        self.__tokenizer__= AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.__model__= AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
        self.__training_args__ = TrainingArguments(exp_name , **training_params)
        self.training_params=training_params
        
                                  
    def tokenize_data(self,text):
        return self.__tokenizer__(text, padding='max_length',truncation=True)
    
    def finetune(self, training_data ,testing_data):
        self.trainer = Trainer(
            model=self.__model__, args=self.__training_args__, train_dataset=training_data,
            
            eval_dataset=testing_data,
        )
        self.trainer.train()
        

    def save_model_tokenizer(self,path):
        self.trainer.save_model(path+"/model")
        self.__tokenizer__.save_pretrained(path+"/tokenizer")
        print(f"saved model and tokenizer : {path}")

    def compute_metrics(self,eval_pred):
        
        load_accuracy = load_metric("accuracy",trust_remote_code=True)
        load_f1 = load_metric("f1",trust_remote_code=True)

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        print("predictions[0]",predictions[0])
        accuracy = load_accuracy.compute(predictions=predictions, references=labels)
        f1 = load_f1.compute(predictions=predictions, references=labels, average="weighted")
        return {"accuracy": accuracy, "f1score": f1}

    def evaluate_primary_test(self):
        self.trainer.compute_metrics=self.compute_metrics
        self.eval_primary_data_report=self.trainer.evaluate()
        return self.eval_primary_data_report
    def evaluate_secondary_test(self,sec_data):
        predictions = self.trainer.predict(sec_data)
        self.eval_secondary_data_report=self.compute_metrics((predictions.predictions,predictions.label_ids))
        return self.eval_secondary_data_report


    

