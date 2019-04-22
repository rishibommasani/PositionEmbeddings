# PositionEmbeddings
This project studies position embeddings with the primary objectives being to understand what these embeddings encode, how to better encode postion as a function of existing embedding methods, and whether embeddings are suitable for stand-alone analysis from the larger models they fit within or as pretrained embeddings in other settings. An underlying motivation is position embeddings for self-attentive architectures where we can consider position embeddings as the key step in permitting a shift from recurrent models that explicitly model sequential structure in language to self-attentive models that do so implicitly. An orthogonal direction that this work may be interpreted with is recovering linguistic understanding from existing position embeddings since it appears both standard linguistic techniques and informal human perception appeals to linguistic notions such as part of speech or dependency structure rather than explicit position (that is one can say a great deal more regarding their understanding of "nouns" and their distribution as opposed to segments that appear in "position 3" of a sequence and their distribution).   

Models considered (* indicates largely in progress):  
BERT  
GPT   
Transformer-XL*
GPT2*

Specific variations of these models are the standard varieties available in the huggingface repository: https://github.com/huggingface/pytorch-pretrained-BERT. A property we exploit is BERT and GPT have a fixed equal-dimensional position space of 512 and embed positions into a 784 dimensional space (Transformer-XL uses relative position and GPT2 uses 1024 positions, hence adjustment needs to be made accordingly.). This means both have position embedding matrices of shape: 512 x 784. Both models use absolute position unlike Transformer-XL which uses relative position. 
  
Tokenization:  
Since tokenization is applied before marking position for words/segments, distinction in tokenization schemes accross models can propogate to position annotation. As a result, in this work we fundamentally assume that we can directly compare the position embedding for model A at position i to the position embedding for model B at position i. We provide a simple analysis and statistics regarding the alignment of segments to help understand the extent to which this assumption can be considered valid (and whether this judgment is position-dependent). Specifically, BERT for English is tokenized with WordPiece tokenization whereas GPT is tokenized with BPE tokenization. 
  
Unsupervised data analysis:  
Motivated by the earlier observation that human understanding of position and general intuition may be lacking compared to linguistic attributes such as POS or dependency tags, we provide t-SNE based visualization. We find there are clear differences in the geometry of the embedding space that are still present in the low-rank visualization produced by t-SNE.   
  
Probing:  
The rest of this repository works towards further probing of position embeddings.   


