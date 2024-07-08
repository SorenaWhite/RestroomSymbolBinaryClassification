# RestroomSymbolBinaryClassification
##### Official implementation of 'MobileMLP-LLM: A Method of Recognizing Restroom Signs'


### Prepare dataset
- MMLRestroomSign Dataset concludes 3312 pair of restroom signs. Download it by:
```

```

- The file structure should be:
```

```

### Train with MobileMLP only
```
bash scripts/train_mobilemlp.sh
```

### Train with MobileMLP+LLM
```
bash scripts/train_mobilemlpllm.sh
```

### Test with MobileMLP or MobileMLP+LLM
```
bash gradio/run_app.py
```
