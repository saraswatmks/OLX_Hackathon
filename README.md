# OLX_Hackathon

This was the second round of Code and Curious Competition organised by OLX in association with HackerRank. <br />

In this round, two problems (NLP and Image Recognition) were given to be solved in a span of 3 hours. <br />
As you guess right, I opted for the NLP problem. <br />

Following is my approach and few insights about the problem :
  * The given problem was a real world case which OLX Berlin Team is solving these days.  
  * The dataset had 4 variables namely: ID, title, description and category. Category was the target variable.
  * The target variable had 55 classes. The evaluation metric was accuracy. 
  * I used R to solve this problem. I used text2vec package for creating features like bigrams. 
  * Feature Engineering: Features such as number of characters, count of words, count of numbers, ratio of number to words
  * Cleaning: Removed stopwords, puntuations, spaces, converted to lowercase. All of this was done using regular expressions.
  * Failed Miserably in Model Training. The dataset was large enough that my laptop couldn't support the processing. 

<br />
Things I learned or  knew before which helped:
  * In NLP problems, start your training with Naive Bayes rather than jumping on straight to xgboost.
  * In case of large data, don't hesitate in sampling your train data. But, make sure class distribution is maintained.
  * Keep your code pipeline ready. I had to write each line of code from scratch.
  * Always write your files once you finish with some computation which took long time, so that you don't have to go through it again.



