from datasets import load_dataset, ClassLabel
dataset = load_dataset('mrzlab630/trading-candles')
# set without context, bullish or bearish output specifically
dataset = dataset.filter(lambda x: len(x['output']) == 7)

def format_answers(example):
  sample = example['output']
  example['output'] = 1 if sample == 'Bullish' else 0
  return example
  
dataset = dataset.map(format_answers)
dataset = dataset['train'].remove_columns('input')
dataset = dataset.rename_column('instruction', 'text')
dataset = dataset.rename_column('output', 'label')
dataset = dataset.cast_column('label', ClassLabel(num_classes=2, names=['Bearish', 'Bullish']))
print(dataset[420])

dataset = dataset.train_test_split(test_size=0.3)
print(dataset)

if __name__ == '__main__':
  dataset.save_to_disk('trading-candles-subset-sc-format')
  dataset.push_to_hub('trading-candles-subset-sc-format')
