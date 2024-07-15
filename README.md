
# PejorativITy: Disambiguating Pejorative Epithets to Improve Misogyny Detection in Italian Tweets

This repository contains the code and resources for the project ["PejorativITy: Disambiguating Pejorative Epithets to Improve Misogyny Detection in Italian Tweets"](https://aclanthology.org/2024.lrec-main.1112/) presented at LREC-COLING 2024.

## Abstract

_Misogyny is often expressed through figurative language. Some neutral words can assume a negative connotation when functioning as pejorative epithets. Disambiguating the meaning of such terms might help the detection of misogyny. In order to address such task, we present PejorativITy, a novel corpus of 1,200 manually annotated Italian tweets for pejorative language at the word level and misogyny at the sentence level. We evaluate the impact of injecting information about disambiguated words into a model targeting misogyny detection. In particular, we explore two different approaches for injection: concatenation of pejorative information and substitution of ambiguous words with univocal terms. Our experimental results, both on our corpus and on two popular benchmarks on Italian tweets, show that both approaches lead to a major classification improvement, indicating that word sense disambiguation is a promising preliminary step for misogyny detection. Furthermore, we investigate LLMs’ understanding of pejorative epithets by means of contextual word embeddings analysis and prompting._

## Data

The ``datasets`` folder contains the pejorativITy corpus.
The list of anchors is in ``anchors.json``. 

## Experiments

All runnable scripts are in ``runnables`` folder.

### Fine-tuning AlBERTo

Run ``training.py``.
Eventually save weights in ``pretrained_weights/alberto_pej``.

### Analyze embeddings

Run ``analyze_pretrained_embedding.py`` to analyze pre-trained AlBERTo contextual embeddings.
Run ``analyze_finetuned_embedding.py`` to analyze fine-tuned AlBERTO contextual embeddings.

### Analyze anchors

Run ``anchor_frequency.py``.

## Contact

Arianna Muti: arianna.muti2@unibo.it

Federico Ruggeri: federico.ruggeri6@unibo.it

## Cite

You can cite our work as follows:

```
@inproceedings{muti-etal-2024-pejorativity-disambiguating,
    title = "{P}ejorativ{IT}y: Disambiguating Pejorative Epithets to Improve Misogyny Detection in {I}talian Tweets",
    author = "Muti, Arianna  and
      Ruggeri, Federico  and
      Toraman, Cagri  and
      Barr{\'o}n-Cede{\~n}o, Alberto  and
      Algherini, Samuel  and
      Musetti, Lorenzo  and
      Ronchi, Silvia  and
      Saretto, Gianmarco  and
      Zapparoli, Caterina",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.1112",
    pages = "12700--12711",
    abstract = "Misogyny is often expressed through figurative language. Some neutral words can assume a negative connotation when functioning as pejorative epithets. Disambiguating the meaning of such terms might help the detection of misogyny. In order to address such task, we present PejorativITy, a novel corpus of 1,200 manually annotated Italian tweets for pejorative language at the word level and misogyny at the sentence level. We evaluate the impact of injecting information about disambiguated words into a model targeting misogyny detection. In particular, we explore two different approaches for injection: concatenation of pejorative information and substitution of ambiguous words with univocal terms. Our experimental results, both on our corpus and on two popular benchmarks on Italian tweets, show that both approaches lead to a major classification improvement, indicating that word sense disambiguation is a promising preliminary step for misogyny detection. Furthermore, we investigate LLMs{'} understanding of pejorative epithets by means of contextual word embeddings analysis and prompting.",
}
```


## License

This project is licensed under CC.BY License.
