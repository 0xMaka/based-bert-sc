# Based Bert for sequence classification
This model is a POC and shouldn't be used for any production task.

## Model description
Based Bert SC is a text classification bot for binary classification of a trading candles opening and closing prices.

## Uses and limitations
This model can reliably return the bullish or bearish status of a candle given the opening, closing, high and low, in a format shown. 
It will have trouble if the order of the numbers change (even if tags are included).

### How to use
You can use this model directly with a pipeline
```python
>>> from transformers import pipeline
>>> pipe = pipeline("text-classification", model="0xMaka/based-bert-sc")
>>> text = "identify candle: open: 21788.19, close: 21900, high: 21965.23, low: 21788.19"
>>> pipe(text)
[{'label': 'Bullish', 'score': 0.9999682903289795}]
```

## Finetuning

This model was fine tuned on a custom dataset (https://huggingface.co/datasets/0xMaka/trading-candles-subset-sc-format), using an RTX-3060-Mobile
```
// BUS_WIDTH = 192
// CLOCK_RATE = 1750 
// DDR_MULTI = 8 // DDR6
// BWTheoretical = (((CLOCK_RATE * (10 ** 6)) * (BUS_WIDTH/8)) * DDR_MULI) / (10 ** 9) 
// BWTheoretical == 336 GB/s
```
Self-measured effective (GB/s): 316.280736
