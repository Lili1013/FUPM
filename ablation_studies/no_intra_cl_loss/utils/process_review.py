import pandas as pd
from loguru import logger
import gzip

def select_reviews(source_path,to_path,items,users):
    g = gzip.open(source_path, 'r')
    review_list = []
    i = 0
    for line in g:
        d = eval(line, {"true": True, "false": False, "null": None})
        if (d['asin'] in items) and (d['reviewerID'] in users):
            if i % 10000 == 0:
                logger.info(i)
            i+=1
            review_list.append([d['reviewerID'], d['asin'],d['reviewText']])
    df = pd.DataFrame(review_list, columns=['user_id', 'item_id','review_text'])  # 转换为dataframe
    df.to_csv(to_path, index=False)

if __name__ == '__main__':
    df = pd.read_csv('../../datasets/phone_sport/phone/phone_inter_new.csv.csv')
    users = df['userID']
    select_reviews(source_path='/data/lwang9/datasets/amazon/review_texts/reviews_Cell_Phones_and_Accessories.json.gz')