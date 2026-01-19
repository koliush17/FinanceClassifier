## Dataset preprocessing
1. Check whether there are any missing values. 
2. Remove unnecessary columns country and currency because they won't be used for classification. 
3. Rename columns as in the task description for clarity
4. Check the names and the number of transaction types. 
5. I reduced the dataset to 4k samples so it is enough to train a good model and it don't take long time to wait while training. 

## Text preprocessing
1. Lowercase all words in a sentence
2. Remove symbols like ' , ! ? # etc.
3. Remove word 'online' as it appears many times and does not give any valuable information. 
4. Remove words inside brackets - usually state how transaction was made, e.g card, cash. 
5. Remove extra whitespaces
6. Remove numbers from string where they are not of the Income type. 
