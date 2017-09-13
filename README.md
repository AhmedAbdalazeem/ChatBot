## CUDA-Dockerized Implementation of Hybrid (Generative and Retrieval) Based Conversational ChatBot Model in TensorFlow.



The current results are pretty lousy:

    hello   	        - hello
    how old are you ?   - twenty .
    i am lonely	        - i am not
    nice                - you ' re not going to be okay .
    so rude	            - i ' m sorry .

**picture**

[![seq2seq](https://4.bp.blogspot.com/-aArS0l1pjHQ/Vjj71pKAaEI/AAAAAAAAAxE/Nvy1FSbD_Vs/s640/2TFstaticgraphic_alt-01.png)](http://4.bp.blogspot.com/-aArS0l1pjHQ/Vjj71pKAaEI/AAAAAAAAAxE/Nvy1FSbD_Vs/s1600/2TFstaticgraphic_alt-01.png)

Curtesy of [this](http://googleresearch.blogspot.ru/2015/11/computer-respond-to-this-email.html) article.

**Setup**

    git clone git@github.com:AhmedAbdalazeem/ChatBot.git
    cd tf_seq2seq_chatbot
    bash setup.sh
    
**Run**

Train a seq2seq model on a small (17 MB) corpus of movie subtitles:

    python train.py
    
(this command will run the training on a CPU... GPU instructions are coming)

Test trained trained model on a set of common questions:

    python test.py
    
Chat with trained model in console:

    python chat.py
    
All configuration params are stored at `tf_seq2seq_chatbot/configs/config.py`

**GPU usage**

If you are lucky to have a proper gpu configuration for tensorflow already, this should do the job:

    python train.py
    
Otherwise you may need to build tensorflow from source and run the code as follows:

    cd tensorflow  # cd to the tensorflow source folder
    cp -r ~/tf_seq2seq_chatbot ./  # copy project's code to tensorflow root
    bazel build -c opt --config=cuda tf_seq2seq_chatbot:train  # build with gpu-enable option
    ./bazel-bin/tf_seq2seq_chatbot/train  # run the built code

**Requirements**

* [tensorflow](https://www.tensorflow.org/versions/master/get_started/os_setup.html)


**References**

* https://github.com/nicolas-ivanov/tf_seq2seq_chatbot
* https://github.com/suriyadeepan/easy_seq2seq
* https://github.com/gunthercox/ChatterBot
* https://people.mpi-sws.org/~cristian/Cornell_Movie-Dialogs_Corpus.html
