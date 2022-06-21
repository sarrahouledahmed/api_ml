from utils import *
from utils import Preprocessing_Google,Preprocessing,Spliter
TARGET_FB_NAME=['result','reach', 'impressions']
TARGET_Google_NAME=['clicks','impressions','views']

### Load FB and Google dataset
data_Google= pd.read_excel('C:/Users/admin/Downloads/google(Clicks+Impressions+Views).xlsx')
data_FB= pd.read_excel('C:/Users/admin/Downloads/Global_Data (3).xlsx')

###########Preprocessing_FB
data_FB=Preprocessing.__Preprocess__(data_FB,TARGET_NAME=TARGET_FB_NAME,type="train")
print(data_FB)
split=Spliter.__Spliter__(data_FB,TARGET_NAME=TARGET_FB_NAME)
TRAIN_FB_DATA=split[0]
TEST_FB_DATA=split[1]
TRAIN_FB_DATA.to_csv(TRAIN_DATA_FB_PATH, index=False)
TEST_FB_DATA.to_csv(TEST_DATA_FB_PATH, index=False)
###########Preprocessing_Google
data_Google=pd.DataFrame(data_Google[['Week', 'Campaign type', 'Currency code', 'Clicks', 'Impr.','Cost','Views']])
df=Preprocessing_Google.__Preprocess__(data_Google,TARGET_NAME=TARGET_Google_NAME,type="train")
split=Spliter.__Spliter__(df,TARGET_NAME=TARGET_Google_NAME)
TRAIN_Google_DATA=split[0]
TEST_Google_DATA=split[1]
TRAIN_Google_DATA.to_csv(TRAIN_DATA_Google_PATH, index=False)
TEST_Google_DATA.to_csv(TEST_DATA_Google_PATH, index=False)