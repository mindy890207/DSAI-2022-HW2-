# DSAI-2022-HW2-
## Objective:
利用歷史資料training_data.csv，來決定接下來20天要如何買賣股，使獲利最多

## Dataset:
1. 助教提供之training_data.csv，含超過五年之daily prices

## Method:
### 預測:
使用LSTM模型將training_data.csv拿去做預測  
並把預測的結果存入predict.csv  
預測結果如下:  
![image](https://user-images.githubusercontent.com/60889705/164981042-072772c5-6cf1-441f-9210-a04955b9240f.png)  
### 計算
使用均線的概念，利用過去幾天的平均成交價格來找出整體趨勢。   
在短期均線高於中期均線時買進，中期均線大於短期均線時賣出。  
其中短期均線與中期均線的天數分別訂為3跟13天，且一開始是用predict.csv中預測的價格來計算。  
而testing_data.csv中的實際價格則會等到當天開盤價出來後，才會更新到predict.csv，以調整均線。   
最後算出的結果共賣空一次，並最後以收盤價買回。   
## Run
python trader.py --training training_data.csv --testing testing_data.csv --output output.csv
