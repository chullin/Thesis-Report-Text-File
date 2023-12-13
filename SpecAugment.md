###### tags: `論文報告文字檔`
# 報告
* [SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
 論文](https://arxiv.org/pdf/1904.08779.pdf)
* [Listen, Attend and Spell 論文](https://arxiv.org/pdf/1508.01211.pdf)
* [機器不學習：理解LSTM/RNN中的Attention機制](https://kknews.cc/zh-tw/code/z5lzl6a.html)
* [對LSTM和BLSTM的一些理解](https://www.twblogs.net/a/5b83ce812b71777cb15c2c0d)
* [Word Piece Model (WPM) 笔记](https://blog.csdn.net/feifei3211/article/details/103053126)
* [遞歸神經網路（RNN）和長短期記憶模型（LSTM）的運作原理](https://brohrer.mcknote.com/zh-Hant/how_machine_learning_works/how_rnns_lstm_work.html)
* [一文讀懂Attention：Facebook曾拿CNN秒殺谷歌，現如今谷歌拿它秒殺所有人](https://kknews.cc/zh-tw/tech/e8e4poz.html)
* [深度學習：CNN原理](https://medium.com/@CinnamonAITaiwan/%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92-cnn%E5%8E%9F%E7%90%86-keras%E5%AF%A6%E7%8F%BE-432fd9ea4935)
* [深度學習基礎--注意力機制(attention)](https://www.itread01.com/content/1545318552.html)
* [一文搞懂NLP中的Attention機制（附詳細程式碼講解](https://www.mdeditor.tw/pl/p0iI/zh-tw)
* [GOOGLE 發布用於自動語音辨識的資料增強新方法](https://goog-book.blogspot.com/2019/04/google.html)
* [簡單粗暴而有效的改圖：自動語音識別數據擴增的「一條野路」](https://kknews.cc/zh-tw/code/qgvppjr.html)
* [效能超越經典 ASR 模型：谷歌重磅推出全新語音識別資料增強方法](https://www.mdeditor.tw/pl/2Kfh/zh-tw)
* [语音识别技术 – ASR丨Automatic Speech Recognition](https://easyai.tech/ai-definition/asr/)
* [谷歌提出新型自動語音識別數據增強大法，直接對頻譜圖「動刀」，提升模型表現](https://read01.com/RMPzQgg.html)
* [图像仿射变换及图像扭曲(Image Warping)](https://blog.csdn.net/ZYTTAE/article/details/42507541)
* [對迴圈神經網路（RNN）中time step的理解](https://www.itread01.com/content/1547198528.html)
* [Getting to Know the Mel Spectrogram](https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0)
* [聲學特徵提取：梅爾頻率倒譜係數MFCC｜PPLOVELL講語音｜5th](https://kknews.cc/tech/z5ng553.html)
* [log 梅爾譜圖的 Google 搜尋](https://www.google.com/search?q=log+%E6%A2%85%E7%88%BE%E8%AD%9C%E5%9C%96&rlz=1C1SQJL_zh-TWTW871TW871&sxsrf=ALeKk00wMsD6XDsJGbQ8hUh-LCTUnDb8bA:1599645742550&source=lnms&tbm=isch&sa=X&ved=2ahUKEwji7YDi6NvrAhV3wosBHVeBBHIQ_AUoAnoECAwQBA&biw=958&bih=927)





![](https://i.imgur.com/QJqWNbU.png)
老師好，我這次要報告的論文是 Google 所提出，對於自動語音辨識資料簡單的擴增方法：SpecAugment

![](https://i.imgur.com/E6tsy4L.png)
我會依序講 介紹、實驗擴增方法、測試模型、實驗結果、討論 還有 結論

![](https://i.imgur.com/LteL4em.png)
ASR 自動語音辨識 是一種將音訊輸入轉換成文字的技術

深度神經網路的發展推動了 ASR 的進步，許多現代裝置與產品都具有其應用，像是 google 助理、Siri、Youtube

![](https://i.imgur.com/G5TUzBo.png)
在本篇論文中，Google 推出了一種新的資料增強方法，叫做 SpecAugment 

而 SpecAugment 與傳統方法有什麼區別呢？

由於傳統語音辨識是使用人工錄製的方式來增加資料模型，因此面臨資料量不足，且錄製需求量大、成本高的問題

且傳統語音辨識包含許多參數，當訓練集不足或不夠全面時，模型往往會產生過度擬合的問題，在新的測試資料上會有更高的錯誤率


而 SpecAugment 採用新的方法來增強音訊資料，將其視為視覺問題，而不是音訊問題
與傳統增加輸入音訊波形不同的是，
它針對音訊頻譜圖做資料增強，因為方法簡單、計算量小，因此成本較低，並且能有效提升 ASR 網路的表現效能

![](https://i.imgur.com/7CPjeYO.png)
接著是實驗擴增方法
SpecAugment 將音訊轉換成 梅爾倒頻譜 後進行圖像處理，
總共有三種方式

![](https://i.imgur.com/kEhiysB.png)
第一種 時間扭曲

![](https://i.imgur.com/VTxIVyl.png)

它透過 tensorflow 的 sparse_image_warp 函式來達成


給一個時間步長為 Tau 的梅爾倒頻譜，將其視為一個圖像，
X軸為時間，Y軸為頻率
對時間軸隨機一點做變形
隨機點向左向右總共距離為W
W 為時間變形參數

上面為原圖，下面為變形後的梅爾頻譜圖

1.空間座標轉換
2.變換座標的賦值運算、插值運算

![](https://i.imgur.com/0KjCi6v.png)
第二種 對頻率做遮罩

![](https://i.imgur.com/KxZl1TA.png)

對 f 個連續頻率 做遮罩 也就是 ( f0, f0+f ) 
其中 f 是從 [ 0, 大F ] 中選擇，大 F 為頻率遮罩參數
而 f0 是從 [ 0, v-f ] 中選擇，v 為頻率通道的數量

下圖是頻率遮罩與原本做比較
而這張圖的 v 就是 1，只使用一個頻率遮罩

![](https://i.imgur.com/TUVjwur.png)
第三種 對時間做遮罩

![](https://i.imgur.com/hLw0jfQ.png)

對 t 個連續時間步長 ( t0, t0+t ) 做遮罩
其中 t 是從 [ 0, 大T ] 中選擇，大T 為時間遮罩參數
而 t0 是從 [ 0, τ-t ] 中選擇

其中，時間遮罩的寬度不能超過時間步長的 p 倍

![](https://i.imgur.com/rIfpNcz.png)
當增強策略包含多個頻率和時間 遮罩 時，多個 遮罩 可能會互相覆蓋。
因此，作者使用四種手動增強的策略：LB、LD、SM 和 SS (用螢光筆畫)
分別為

LibriSpeech basic、 double，
Switchboard mild 輕度 和
 strong 強烈

mF和mT 表示頻率和時間遮罩的使用數量

![](https://i.imgur.com/CbxhV41.png)

這是 LB 和 LD 與原圖得比較

![](https://i.imgur.com/x1x6EDr.png)

接著是測試模型

網路架構是使用 (LAS) Listen, Attend and Spell 作為網路模型，LAS主要是由 Listener 和 Speller 兩個子模型組成
Listener 是聲學編碼器 (相當於聽)、
而Speller 是基於記憶力機制的解碼器 (相當於說)


input 是 梅爾倒頻譜
經過 CNN(捲積網路) 且 Max-pooling 後，再輸入進BLSTM(雙向的長短期記憶模型) 輸出 attention vector

之後再將 attention vector 輸出進 RNN(遞迴神經網路) 裡
最後 output 是字符

CNN (捲積網路)
Max-pooling (最大池化)
BLSTM (雙向的長短期記憶模型)
RNN (遞迴神經網路)

Max Pooling 只要挑出矩陣當中的最大值，好處是有很好的抗雜訊功能。

BLSTM 會產生注意力向量，就是一個有權重的vector

Attention 在 每生成一個詞時都會在輸入序列中找出一個與之最相關的詞集合。之後模型根據當前的上下文向量 (context vectors) 和所有之前生成出的詞來預測下一個目標詞。





![](https://i.imgur.com/L6oVKAH.png)

學習率是 ASR 網路表現的一個重要因素
論文中研究學習率有兩個目標，
一是驗證較長的策略有助於提升網路最終的表現
二是如果較長的策略能有效提升那就引入一個更長的學習策略，來最大化網路的表現效能

實驗中，這三個階段定義為 ( sr提升, si維持, sf指數衰減)
另外還新增其他參數來影響學習率，像是噪音


論文中使用的兩個基本方法為
1. B(asic): (sr, snoise, si, sf)=(0.5k, 10k, 20k, 80k)
2. D(ouble): (sr, snoise, si, sf)=(1k, 20k, 40k, 160k)

另外使用更長的時間策略(Long)來最大化網路表現
3. L(ong):(sr, snoise, si, sf)=(1k, 20k, 140k, 320k)


![](https://i.imgur.com/gFCbsLN.png)
另外訓練集中的文本透過 WPM (Word Piece Model) 進行 token 處理，並利用 集束搜尋 (Beam Search) 得到預測文本

 token 處理 即分詞處理，將一個漢字序列切分成一個一個單獨的詞
 之後進行詞嵌入，自然語言處理利用詞嵌入來將詞向量化表示

至此實現語音轉文本的過程

![](https://i.imgur.com/PRqnx4C.png)

接著是實驗結果

在 LibriSpeech 960h 的表現結果
論文中使用了三種 LAS 網路作為 ASR 模型
分別為 
LAS-4-1024、LAS-6-1024 和 LAS-6-1280

d 個堆疊的雙向LSTM
w 是一系列的注意力向量
簡單來說，attention就是一個有權重的vector


然後 sch 是指
訓練策略
learning rate schedules 

pol 是指
增強策略
augmentation policies 

並利用訓練策略和增強策略的組合在 LibriSpeech 資料集上訓練。


其中只改變 訓練策略 sch 與 增強策略 pol

由此圖可以證明更大的模型和更長的學習策略可以有效增強訓練結果

每個比較都是錯誤率越來越小
像是比較每一個網路由上到下數字會越來越小

最右下角，10.5 > 7.1 > 6.5

![](https://i.imgur.com/zn1e7HV.png)

接著是 Switchboard 300h 的表現結果
論文中使用一個網路架構 LAS-4-1024 和使用一種訓練策略  Basic

改變的只有增強策略 augmentation policies 
分為(不使用、Switchboard mild 和Switchboard strong


LS 為 Label Smoothing 平滑標籤，
從下圖可以看出來標籤平滑在學習率較小時會影響網路的穩定性

可以看到 有沒有使用 LS 平滑標籤 和 SM、SS 皆會影響網路效能

![](https://i.imgur.com/h4YX21U.png)
接著是作者針對此論文做的討論，分為 3 點


時間變形有用，但不是提升表現的主要原因
	時間變形作為影響最小，但成本最高的策略，應該首先被丟棄。
2. 資料增強使過擬合問題轉化為欠擬合問題
	SpecAugment 通過故意提供損壞的資料來防止網路過度擬合， 
	下圖可以看出如果不進行資料增強，在訓練集上的效能近乎完美，這個 None 棕色的線，但在測試集上效能卻打則扣
	另一方面，如果先將資料進行處理，在訓練集上很難取得完美，但在測試集上卻能表現的更好
3. 常用的欠擬合問題解決方法可以提升網路效能
	解決欠擬合有兩個標準方法：更大的網路和更長的訓練時間，從而在表現效能上取得顯著的進步。

![](https://i.imgur.com/IDjft7B.png)
SpecAugment 大大的提升了 ASR 網路的效能
通過簡單的策略，增加訓練集，即使沒有語言模型的幫助，也能讓 LAS 網路在 LibriSpeech、SwitchBoard 語料庫上獲得更佳的表現
