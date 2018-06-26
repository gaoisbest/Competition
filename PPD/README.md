# Solution of question pair match problem of [PPD](https://ai.ppdai.com/mirror/goToMirrorDetail?mirrorId=1)
to do...

# Features
- Simple statistics, such as the length (and difference in lengths) of the pairs of questions;
- Similarities between the pairs based on the characters constituting the questions,such as Levenshtein and related similarities used in fuzzy string matching;
- Similarities between the pairs based on the set of words constituting the questions, such as the Jaccard index;
- Various distances (L1, L2, cosine, canberra…) between vector representationsof the questions derived from TF-IDF statistics and pre-trained Word2Vec models (Google News, Glove, FastText);

# References
- [Distance and angle features of two lstm states](https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning)
- https://techblog.cdiscount.com/participation-kaggle-challenge-quora-question-pairs/
- [BiMPM](https://arxiv.org/pdf/1702.03814.pdf)
- https://towardsdatascience.com/identifying-duplicate-questions-on-quora-top-12-on-kaggle-4c1cf93f1c30
