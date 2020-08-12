## Welcome to GitHub Pages

## 20200811 commit




## 20200807 commit
* Neural Machine Translation of Rare Words with Subword Units [ACL 2016] [[__pdf__](https://arxiv.org/pdf/1508.07909.pdf)]  [[__code__](https://github.com/rsennrich/subword-nmt)] 
    - Motivation: Neural machine translation (NMT) models typically operate with a fixed vocabulary, but translation is an open-vocabulary problem.
    - Contribution: In this paper, we introduce a simpler and more effective approach, making the NMT model capable of open-vocabulary translation by encoding rare and unknown words as sequences of subword units. 
    - Approach: We adapt byte pair encoding (BPE) (Gage, 1994), a compression algorithm, to the task of word segmentation.
    - Experient: improve over a back-off dictionary baseline for the WMT 15 translation tasks English→German and English→Russian by up to 1.1 and 1.3 BLEU, respectively.
    - Conclusion: The main contribution of this paper is that we show that neural machine translation systems are capable of open-vocabulary translation by representing rare and unseen words as a sequence of subword units.
    - Comment: 著名的BPE算法，作者给了实现的代码，simple but effective，证明了在NMT里subword有效，后续大家也都采用这一方法
    

* Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks [ACL 2020] [[__pdf__](https://arxiv.org/pdf/2004.10964.pdf)]  [[__code__](https://github.com/allenai/dont-stop-pretraining)] 
    - Motivation: In light of the success of these broad-coverage models, we investigate whether it is still helpful to tailor a pretrained model to the domain of a target task.
    - Contribution:  
        1. a thorough analysis of domain- and task-adaptive pretraining across four domains and
    eight tasks, spanning low- and high-resource settings;
        2. an investigation into the transferability of
    adapted LMs across domains and tasks; and
        3. a study highlighting the importance of pretraining on human-curated datasets, and a simple data selection strategy to automatically
    approach this performance.
    - Approach: domain-adaptive pretraining（DAPT) continued pretraining on the domain consistently improves performance on tasks from the target domain, in both high- and low-resource settings. We study how domain-adaptive pretraining compares to task-adaptive pretraining, or TAPT, on a smaller but directly task-relevant corpus.
    - Experient: DAPT / TAPT
    - Conclusion: We show that pretraining the model towards a spe- cific task or small corpus can provide significant benefits. Our work points to numerous future directions, such as better data selection for TAPT, efficient adaptation large pretrained language models to distant domains, and building reusable language models after adaptation.
    - Comment: Domain-Adaptive PreTraining (DAPT)是说在大规模的无监督domain语料上预训练。 Task-Adaptive PreTraining (TAPT) 是说在下游任务相关的无监督小语料上进行预训练。
        这篇工作证明了几个结论：
        1. 在无监督domain语料上继续预训练和下游任务相关的无监督小语料上继续预训练都能得到很大提升。
        2. DAPT实验中，首先检查了四个domain和源domain（roberta训练语料）之间的相似度，经过实验证明了，domain和源domain相差越大，DAPT提升越明显。the more dissimilar the domain, the higher the potential for DAPT.
        3. 非常重要的是，做了一个data selection的实验，证明了select data有用。并且提出未来可以探究better data selection, efficient adaptation strategy, 这个正是我们想要做的。作者提出未来可以用更好的data selection的策略进行研究。
        TAPT的好处是，it uses a far smaller pretraining corpus, but one that is much more task-relevant. This makes TAPT much less expensive to run than DAPT. 从实验结果看，DAPT和TAPT的performance各有千秋，但是TAPT更加cheaper。 作者最后把他俩叠加到一起实现了更好的效果。 

* Using Similarity Measures to Select Pretraining Data for NER [NAACL 2019] [[__pdf__](https://arxiv.org/pdf/1904.00585.pdf)]  [[__code__](https://github.com/daixiangau/naacl2019-select-pretraining-data-for-ner)] 
    - Motivation: The measure and impact of similarity between pretraining data and target task data are left to intuition. Our overarching goal is to develop a cost-effective approach that, given a NER data set, nominates the most suitable source data to pretrain word vectors or LMs from several options. Our approach builds on the hypothesis that the more similar the source data is to the target data, the better the pretrained models are, all other aspects (such as source data size) being equal. We propose using target vocabulary covered rate and language model perplexity to select pretraining data. We also introduce a new measure based on the change from word vectors pretrained on source data to word vectors initialized from source data and then trained on target data.
    - Contribution: 
        1. We propose methods to quantitatively measure different aspects of similarity between source and target data sets and find that these
        measures are predictive of the impact of pretraining data on final accuracy. To the best of our knowledge, this is the first systematic
        study to investigate LMs pretrained on various data sources.
        2. We find that it is important to consider tenor as well as field when selecting pretraining data, contrary to human intuitions.
        3. We show that models pretrained on a modest amount of similar data outperform pretrained models that take weeks to train over very large generic data.
    - Approach: Three similarity measures: 1. Target Vocabulary Covered  2. Language Model Perplexity  3. Word Vector Variance
    - Experient: 5 source datasets, 6 target datasets about NER.
    - Conclusion: We investigated how these measures correlate with the effectiveness of pretrained word vectors and LMs for NER tasks. We found that the effectiveness of pretrained word vectors strongly depends on whether the source data have a high vocabulary intersection with target data, while pretrained LMs can gain more benefits from a similar source.
    - Comment: 关于训练的source corpus和target task data的相似度 对于训练结果的影响，以前只是停留在intuition的阶段。本文首先找了30个专业NLP/ML人员来给出自己的判断，S1/S2对于T哪个更有效。专家的意见众说纷纭(见figure 1)， 作者提出了三种cost-effective measures to quantify different aspects of similarity between source and target data. 实验证明了作者测量出来的similarity是模型performance好坏的一个predictor。
    我们可以重点借鉴similarity measure来做我们自己的事情
    Important reference：They also investigate the impact of the source data size and find that larger pretraining data do not necessarily produce better word vectors for biomedical NER (Chiu et al. (2016))
    Joshi et al. (2018) empirically showed that, for their vaccination behaviour detection task on twitter data, LMs pretrained on a small amount of movie reviews outperform the ones pretrained on large size of Wikipedia data.


* Curriculum Learning for Domain Adaptation in Neural Machine Translation [NAACL 2019] [[__pdf__](https://arxiv.org/pdf/1905.05816.pdf)]  [[__code__](https://github.com/kevinduh/sockeye-recipes/tree/master/egs/curriculum)] 
    - Motivation: We introduce a curriculum learning approach to adapt generic neural machine translation models to a specific domain
    - Contribution: A practical curriculum learning method should address two main questions: how to rank the training examples, and how to modify the sampling procedure based on this ranking. Inspired by curriculum learning (Bengio et al., 2009), we use the similarity scores given by data selection to rearrange the order of training samples, such that more similar examples are seen earlier and more frequently during training. To the best of our knowledge, this is the first work applying curriculum learning to domain adaptation.
    - Approach: We examine two data selection methods, Moore-Lewis method (Moore and Lewis, 2010) and cynical data selection (Axelrod, 2017).
    - Experient: NMT tasks.  We show the effectiveness of our method on four tasks. Results show that curriculum learning models can improve over the standard continued training model by up to 3.22 BLEU points and can take better advantage of distant and noisy data.
    - Conclusion: Our approach first ranks unlabeled- domain training samples based on their similarity to in-domain data, and then adopts a probabilistic curriculum learning strategy so that more similar samples are used earlier and more frequently dur- ing training.
    - Comment:
    这篇文章是第一个做curriculum domain adaptation, 通俗地讲，就是利用target domain data和source domain data的相似度，排一个序。怎么算相似度？两种方法：Moore-Lewis method和cynical data selection， 怎么排序？根据相似度，具有相似的相似度的sentence分到同一个shard里面。 排好序之后，怎么用？probabilistic curriculum training 在同一个shard里面的data会ramdom sample batch，同一个phase的不同shard之间会random一下。


* Unsupervised Domain Clusters in Pretrained Language Models [ACL 2020] [[__pdf__](https://arxiv.org/pdf/2004.02105.pdf)]  [[__code__](https://github.com/roeeaharoni/unsupervised-domain-clusters)] 
    - Motivation: The notion of “in-domain data” in NLP is often over-simplistic and vague, as textual data varies in many nuanced linguistic aspects such as topic, style or level of formality. In addition, domain labels are many times unavailable, making it challenging to build domain-specific systems. We show that massive pre-trained language models implicitly learn sentence representations that cluster by domains without supervision–suggesting a simple data-driven definition of domains in textual data
    - Contribution: 
        1. First, we show that pre-trained language models are highly capable of clustering textual data to domains with high accuracy in a purely unsupervised manner. 
        2. Second, we propose methods to select in-domain data based on this property using vector-space retrieval and positive-unlabeled fine-tuning of pretrained language models for binary classification. 
        3. Third, we show the applicability of our proposed data selection methods on a popular benchmark for domain adaptation in machine translation. 
        4. An additional contribution is a new, improved data split we create for this benchmark, as we point on issues with previous splits used in the literature.
    - Approach: pretrained model + GMM
    - Experient: NMT tasks. We evaluate our method on data selection for neural machine translation (NMT) using the multi-domain German-English parallel corpus composed by Koehn and Knowles (2017). Our data selection methods enable to train NMT models that outperform those trained using the well-established cross-entropy difference method of Moore and Lewis (2010) across five diverse domains, achieving a recall of more than 95% in all cases with respect to an oracle that selects the “true” in-domain data.
    - Conclusion: We showed that massive pre-trained language models are highly effective in mapping data to domains in a fully-unsupervised manner using average-pooled sentence representations and GMM-based clustering.
    - Comment:
    本文做的是，想用pretrain model去挑数据。尝试先用pretrain model将数据无监督地聚类。pretrain model有bert/roberta/gpt等等，聚类的方法是GMM。为什么不用K-means？因为GMM allows soft assignments (vs. hard assignments as in e.g. K-means) which we think fits the task better (as a sentence can be seen as drawn from a mixture of several domain)
    **以下是一段精彩的关于domain的论述。**
    The definition of domain is many times vague and over-simplistic (e.g. “medical text” may be used for biomedical research papers and for clinical conversations between doctors and patients, although the two vary greatly in topic, formality etc.). A common definition treats a domain as a data source: “a domain is defined by a corpus from a specific source, and may differ from other domains in topic, genre, style, level of formality, etc.” (Koehn and Knowles, 2017). We claim that a more data-driven definition should take place, as different data sources may have sentences with similar traits and vice versa - a single massive web-crawled corpus contains texts in numerous styles, topics and registers. Our analysis in Section 2 shows examples for such cases, e.g. a sentence discussing “Viruses and virus-like organisms” in a legal corpus.
    
* To Annotate or Not? Predicting Performance Drop under Domain Shift  [EMNLP 2019] [[__pdf__](https://www.aclweb.org/anthology/D19-1222.pdf)]  [[__code__](https://github.com/hadyelsahar/domain-shift-prediction)] 
    - Motivation: Performance drop due to domain-shift is an endemic problem for NLP models in production. This problem creates an urge to continuously annotate evaluation datasets to measure the expected drop in the model performance which can be prohibitively expensive and slow.In this paper, we study the problem of predicting the performance drop of modern NLP models under domain-shift, in the absence of any target domain labels. We investigate three families of methods (H-divergence, reverse classification accuracy and confidence measures), show how they can be used to predict the performance drop and study their robustness to adversarial domain-shifts. 
    - Contribution: 
        1. We introduce a new task and methodology for directly predicting performance drop of a model under domain-shift, without the need of labeled examples from the target domain.
        2. We survey, formalize and evaluate domain-shift detection metrics from 3 different families (§2) and propose new adaptations.
        3. We benchmark each proposes metric on two tasks of different natures: document classification and sequence labeling (§3 and §4), and show their robustness under adversarial domain-shift scenarios.
    - Approach: three families of methods (H-divergence, reverse classification accuracy and confidence measures)
    - Experient: document classification and sequence labeling
    - Conclusion: Our results on sentiment classification and sequence labelling show that our method is able to predict performance drops with an error rate as low as 2.15% and 0.89% for sentiment analysis and POS tagging respectively.
    - Comment: 
    本文的核心是预测domain shift带来的performance drop有多大。首先，domain shift会带来performance drop是毫无疑问的公认事实，本文的方法是做一个regression，来预测domian-shift detection metrics 和drop之间的关系。metrics有三种：H-divergence, reverse classification accuracy and confidence measures.
    精彩论述, worth to refer：
    有论据证明，即使一个模型在test data上效果很好，也会因为domain shift（比如vocab和style不同的问题），从而造成performance drop。It is well known that modern machine-learning models can be brittle, meaning that – even when achieving impressive performance on the evaluation set – their performance can degrade significantly when exposed to new examples with differences in vocabulary and writing style (Blitzer and Pereira, 2007; Jia and Liang, 2017; Brun and Nikoulina, 2018). 


## 20200804 first commit

### Domain Adaptation
* Active Adversarial Domain Adaptation [WACV 2020] [[__pdf__](https://arxiv.org/pdf/1904.07848.pdf)]
* BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning [NeurIPS 2019] [[__pdf__](https://arxiv.org/pdf/1906.08158.pdf)]  [[__code__](https://github.com/BlackHC/BatchBALD)] 


### Sparse Transformer
* Deep High-Resolution Representation Learning for Visual Recognition [TPAMI 2019] [[__pdf__](https://arxiv.org/pdf/1908.07919.pdf)]  [[__code__](https://github.com/HRNet)] 
* Deep High-Resolution Representation Learning for Human Pose Estimation [CVPR 2019] [[__pdf__](https://arxiv.org/pdf/1902.09212.pdf)]  [[__code__](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)] 
Gradually add high-to-low resolution subnetworks one by one to form more stages, and connect the mutliresolution subnetworks in parallel.
* Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks [NIPS 2015] [[__pdf__](https://arxiv.org/pdf/1506.05751.pdf)]  [[__code__](https://github.com/witnessai/LAPGAN)]  
Use a cascade of convolutional networks within a Laplacian pyramid framework to generate images in a coarse-to-fine fashion.
* Feature Pyramid Networks for Object Detection [CVPR 2017] [[__pdf__](https://arxiv.org/pdf/1612.03144.pdf)] [[__code__](https://github.com/jwyang/fpn.pytorch)]  
Exploit the inherent multi-scale, pyramidal hierarchy of deep convolutional networks to construct feature pyramids with marginal extra cost.
* U-Net: Convolutional Networks for Biomedical Image Segmentation  [MICCAI 2015] [[__pdf__](https://arxiv.org/pdf/1505.04597.pdf)] [[__code__](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net)]  
The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization.
* Fully Convolutional Networks for Semantic Segmentation [CVPR 2015] [[__pdf__](https://arxiv.org/pdf/1411.4038.pdf)] [[__code__](https://github.com/BVLC/caffe/wiki/Model-Zoo#fcn)]  
Build “fully convolutional” networks that take input of arbitrary size and produce correspondingly-sized output with efficient inference and learning.
* Multiresolution Transformer Networks: Recurrence is Not Essential for Modeling Hierarchical Structure  [arXiv Aug 2019] [[__pdf__](https://arxiv.org/pdf/1908.10408.pdf)]
Establish connections between the dynamics in Transformer and recurrent networks to argue that several factors including gradient flow along an ensemble of multiple weakly dependent paths play a paramount role in the success of Transformer. Then leverage the dynamics to introduce Multiresolution Transformer Networks as the first architecture that exploits hierarchical structure in data via self-attention.
* MUSE: Parallel Multi-Scale Attention for Sequence to Sequence Learning [arXiv Nov 2019] [[__pdf__](https://arxiv.org/pdf/1911.09483.pdf)]  [[__code__](https://github.com/lancopku/Prime)]  
Explore parallel multi-scale representation learning on sequence data, striving to capture both long-range and short-range language structures.
* Multi-scale Transformer Language Models [arXiv 1st May 2020] [[__pdf__](https://arxiv.org/pdf/2005.00581.pdf)]
Learn representations of text at multiple scales and present three different architectures that have an inductive bias to handle the hierarchical nature of language.
---------------------------------------

* REFORMER: The Efficient Transformer [ICLR 2020] [[__pdf__](https://arxiv.org/pdf/2001.04451.pdf)] [[__code__](https://github.com/google/trax/tree/master/trax/models/reformer)]  
Replace dot-product attention by one that uses locality-sensitive
hashing, changing its complexity from O(L^2) to O(LlogL), where L is the length of the sequence.
* Longformer: The Long-Document Transformer
 [arXiv Apr. 2020] [[__pdf__](https://arxiv.org/pdf/2004.05150.pdf)] [[__code__](https://github.com/allenai/longformer)]  
Propose an attention mechanism with a drop-in replacement
for the standard self-attention and combines
a local windowed attention with a task motivated global attention.
* Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing [arXiv Jun. 2020] [[__pdf__](https://arxiv.org/pdf/2006.03236.pdf)] [[__code__](https://github.com/laiguokun/Funnel-Transformer)]  
* Lite Transformer with Long-Short Range Attention [ICLR 2020] [[__pdf__](https://arxiv.org/pdf/2004.11886.pdf)] [[__code__](https://github.com/mit-han-lab/lite-transformer)]  
Present an efficient mobile NLP architecture, Lite Transformer, to
facilitate deploying mobile NLP applications on edge devices.
* Tree-structured Attention with Hierarchical Accumulation [ICLR 2020] [[__pdf__](https://arxiv.org/pdf/2002.08046.pdf)] [[__code__](https://github.com/nxphi47/tree_transformer)]  
Present an attention-based hierarchical encoding method.
* Transformer-XH: Multi-Evidence Reasoning with eXtra Hop Attention [ICLR 2020] [[__pdf__](https://openreview.net/pdf?id=r1eIiCNYwS)]
Present Transformer-XH, which uses extra hop attention to enable intrinsic modeling of structured texts in a fully data-driven way.
* Linformer: Self-Attention with Linear Complexity [arXiv Jun. 2020] [[__pdf__](https://arxiv.org/pdf/2006.04768.pdf)] 
Demonstrate that the self-attention mechanism can be approximated by a low-rank matrix and propose a new self-attention mechanism, which reduces the overall self-attention complexity from O(n^2) to O(n) in both time and space.
* Adaptive Attention Span in Transformers [ACL 2019] [[__pdf__](https://arxiv.org/pdf/1905.07799.pdf)] [[__code__](https://github.com/facebookresearch/adaptive-span)] 
Propose a novel self-attention mechanism
that can learn its optimal attention span.
* BP-Transformer: Modelling Long-Range Context via Binary Partitioning
 [arXiv Nov. 2019] [[__pdf__](https://arxiv.org/pdf/1911.04070.pdf)] [[__code__](https://github.com/yzh119/BPT)]  
Adopt a fine-to-coarse attention mechanism on multi-scale spans via binary partitioning.
* Adaptively Sparse Transformers [EMNLP 2019] [[__pdf__](https://arxiv.org/pdf/1909.00015.pdf)] [[__code__](https://github.com/deep-spin/entmax)]  
Introduce the adaptively sparse Transformer,
wherein attention heads have flexible, contextdependent sparsity patterns. This sparsity is accomplished by replacing softmax with α-entmax: a differentiable generalization of softmax that allows low-scoring words to receive precisely zero weight.





You can use the [editor on GitHub](https://github.com/shizhediao/Paper_Reading/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/shizhediao/Paper_Reading/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
