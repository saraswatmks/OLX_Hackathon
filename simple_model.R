path <- "olx_hackathon/"
setwd(path)

# load data and lirbaries -------------------------------------------------
rm(list=ls())
library(data.table)

train <- fread("nlp-dataset/train.csv")
test <- fread("nlp-dataset/test.csv")



# sample data -------------------------------------------------------------
library(caret)

split <- createDataPartition(y = train$category, p = 0.5, list = F)

train <- train[!split]

# target ------------------------------------------------------------------
train[,.N,category] ## 55 categories

# Create Features ---------------------------------------------------------
library(stringr)
train[,title_num := unlist(lapply(title, function(x)  str_count(string = x, pattern = "\\d+")))]
train[,word_title := unlist(lapply(title, function(x) str_count(string = x, pattern = "\\s+") + 1))]
train[,char_title := unlist(lapply(title, function(x) nchar(x)))]
train[,brac_title := unlist(lapply(title, function(x) str_count(string = x, pattern = "\\(|\\)|\\/")))]
train[,cap_count := unlist(lapply(title, function(x) str_count(string = x, pattern = "[A-Z]")))]
train[,deci_count := unlist(lapply(title, function(x) str_count(string = x, pattern = "\\.")))]
train[,digit_num_ratio := title_num/word_title]

test[,title_num := unlist(lapply(title, function(x)  str_count(string = x, pattern = "\\d+")))]
test[,word_title := unlist(lapply(title, function(x) str_count(string = x, pattern = "\\s+") + 1))]
test[,char_title := unlist(lapply(title, function(x) nchar(x)))]
test[,brac_title := unlist(lapply(title, function(x) str_count(string = x, pattern = "\\(|\\)|\\/")))]
test[,cap_count := unlist(lapply(title, function(x) str_count(string = x, pattern = "[A-Z]")))]
test[,deci_count := unlist(lapply(title, function(x) str_count(string = x, pattern = "\\.")))]
test[,digit_num_ratio := title_num/word_title]

train[,desc_num := unlist(lapply(description, function(x)  str_count(string = x, pattern = "\\d+")))]
train[,desc_word := unlist(lapply(description, function(x) str_count(string = x, pattern = "\\s+") + 1))]
train[,desc_char := unlist(lapply(description, function(x) nchar(x)))]
train[,desc_brac := unlist(lapply(description, function(x) str_count(string = x, pattern = "\\(|\\)|\\/")))]
train[,desc_cap_count := unlist(lapply(description, function(x) str_count(string = x, pattern = "[A-Z]")))]
train[,desc_deci_count := unlist(lapply(description, function(x) str_count(string = x, pattern = "\\.")))]
train[,desc_digit_ratio := desc_num / desc_word]

test[,desc_num := unlist(lapply(description, function(x) str_count(string = x, pattern = "\\d+")))]
test[,desc_word := unlist(lapply(description, function(x) str_count(string = x, pattern = "\\s+") + 1))]
test[,desc_char := unlist(lapply(description, function(x) nchar(x)))]
test[,desc_brac := unlist(lapply(description, function(x) str_count(string = x, pattern = "\\(|\\)|\\/")))]
test[,desc_cap_count := unlist(lapply(description, function(x) str_count(string = x, pattern = "[A-Z]")))]
test[,desc_deci_count := unlist(lapply(description, function(x) str_count(string = x, pattern = "\\.")))]
test[,desc_digit_ratio := desc_num / desc_word]

fwrite(train, "train_with_feats.csv")
fwrite(test, "test_with_feats.csv")

# remove stopwords, convert to lowercase, punctuation marks ---------------
library(tidytext)

'%nin%' <- Negate('%in%')
rmvstp <- function(x) {
  t <- unlist(strsplit(x, " ")) 
  t[t %nin%stop_words$word]
}

train[,title := unlist(lapply(title, tolower))]
train[,title := gsub(pattern = "[[:punct:]]",replacement = "", x = title)]
train[,title := gsub(pattern = "[[:cntrl:]]",replacement = "", x = title)]
train[,title := gsub(pattern = "\\d+",replacement = "", x = title)]
train[,title := lapply(title, function(x) strsplit(x, " "))]
train[,title := lapply(title, unlist, use.names=F)]
train[,title := lapply(title, function(x) x[nchar(x) > 2])]

test[,title := unlist(lapply(title, tolower))]
test[,title := gsub(pattern = "[[:punct:]]",replacement = "", x = title)]
test[,title := gsub(pattern = "[[:cntrl:]]",replacement = "", x = title)]
test[,title := gsub(pattern = "\\d+",replacement = "", x = title)]
test[,title := lapply(title, function(x) strsplit(x, " "))]
test[,title := lapply(title, unlist, use.names=F)]
test[,title := lapply(title, function(x) x[nchar(x) > 2])]

train[,description := unlist(lapply(description, tolower))]
train[,description := gsub(pattern = "[[:punct:]]",replacement = "", x = description)]
train[,description := gsub(pattern = "[[:cntrl:]]",replacement = "", x = description)]
train[,description := gsub(pattern = "\\d+",replacement = "", x = description)]
train[,description := lapply(description, function(x) strsplit(x, " "))]
train[,description := lapply(description, unlist, use.names=F)]
train[,description := lapply(description, function(x) x[nchar(x) > 2])]

test[,description := unlist(lapply(description, tolower))]
test[,description := gsub(pattern = "[[:punct:]]",replacement = "", x = description)]
test[,description := gsub(pattern = "[[:cntrl:]]",replacement = "", x = description)]
test[,description := gsub(pattern = "\\d+",replacement = "", x = description)]
test[,description := lapply(description, function(x) strsplit(x, " "))]
test[,description := lapply(description, unlist, use.names=F)]
test[,description := lapply(description, function(x) x[nchar(x) > 2])]


# rbind the data and create corpus
library(text2vec)

tok_fun <- word_tokenizer
all_data <- rbind(train[,.(id, title, description)], test[,.(id,title, description)])

all_data_title = itoken(all_data$title, tokenizer = tok_fun, ids = all_data$id)
all_data_description <- itoken(all_data$description, tokenizer = tok_fun, ids = all_data$id)

#1,2 gram
vocab_alldata_title <- create_vocabulary(all_data_title, ngram = c(1L,2L),stopwords = stop_words$word)
vocab_alldata_title <- vocab_alldata_title %>% prune_vocabulary(term_count_min = 500, doc_proportion_max = 0.3)

vocab_alldata_description <- create_vocabulary(all_data_description, ngram = c(1L,2L),stopwords = stop_words$word)
vocab_alldata_description <- vocab_alldata_description %>% prune_vocabulary(term_count_min = 1000, doc_proportion_max = 0.3)

vec_title <- vocab_vectorizer(vocab_alldata_title)
vec_desc <- vocab_vectorizer(vocab_alldata_description)

all_data_title <- create_dtm(all_data_title, vec_title)
all_data_description <- create_dtm(all_data_description, vec_desc)


# create train and test data
all_title <- as.data.table(as.matrix(all_data_title))
all_desc <- as.data.table(as.matrix(all_data_description))

tr_title <- all_title[1:nrow(train)]
tr_desc <- all_desc[1:nrow(train)]

te_title <- all_title[(nrow(train)+1):nrow(all_title)]
te_desc <- all_desc[(nrow(train) + 1) : nrow(all_title)]

X_train <- cbind(tr_title, tr_desc)
X_test <- cbind(te_title, te_desc)

fwrite(X_train, "train_bag.csv")
fwrite(X_test, "test_bag.csv")

rm(list=setdiff(ls(), c("X_train","X_test","train","test")))

## add features from above files
trainfeat <- fread("train_with_feats.csv")
testfeat <- fread("test_with_feats.csv")

X_train <- cbind(trainfeat, X_train)
X_test <- cbind(testfeat, X_test)

fwrite(X_train, "full_train.csv")
fwrite(X_test, "full_test.csv")

## modeling
## failed
library(h2o)

h2o.init(nthreads = -1, max_mem_size = "20G")

target <- "category"
predictors <- setdiff(colnames(X_train), c(target,"id","title","description"))

train.hex <- as.h2o(X_train)
test.hex <- as.h2o(X_test)

gbm1 <- h2o.gbm()


## try lightgbm
## failed
library(xgboost)

target_map <- data.table(category = unique(X_train$category))
target_map[, maps := .I-1]

X_train <- target_map[X_train, on='category']

predictors <- setdiff(colnames(X_train), c('id','title','description','category','maps'))

set1 <- intersect(colnames(X_train), colnames(X_test))
set1 <- setdiff(set1, c('id','title','description'))

# dtrain <- xgb.DMatrix(data = as.matrix(X_train[,set1,with=F]), label = X_train$maps)
# dtest <- xgb.DMatrix(data = as.matrix(X_test[,set1,with=F]))



# load data
library(data.table)
library(caret)

train1 <- fread("train_with_feats.csv")
test1 <- fread("test_with_feats.csv")

train2 <- fread("train_bag.csv")
test2 <- fread("test_bag.csv")

X_train <- cbind(train1, train2)
X_test <- cbind(test1, test2)

rm(train1,test1,train2, test2)

samp <- createDataPartition(y = X_train$category,p = 0.5,list=F)

X_train <- X_train[!samp]

#
set1 <- intersect(colnames(X_train), colnames(X_test))
set1 <- setdiff(set1, c('id','title','description','category'))

target_map <- data.table(category = unique(X_train$category))
target_map[, maps := .I-1]

X_train <- target_map[X_train, on='category']

set1 <- setdiff(set1, c('maps'))

library(xgboost)

dtrain <- xgb.DMatrix(data = as.matrix(X_train[,set1,with=F]), label = X_train$maps)
dtest <- xgb.DMatrix(data = as.matrix(X_test[,set1,with=F]))


params <- list(
  
  objective = 'multi:softmax',
  num_class = 55,
  eval_metric = 'merror',
  max_depth = 8,
  gamma = 0,
  colsample_bytree = 0.3,
  subsample = 1,
  eta = 0.2,
  min_child_weight = 0
  
)

model1 <- xgb.train(params = params, data = dtrain, nrounds = 50, watchlist = list(train = dtrain))



# add TF-IDF Features
tfidf = TfIdf$new()

dtm_train_tfidf <- fit_transform(dtm_train, tfidf)
dtm_test_tfidf <- create_dtm(it_test, vectorizer) %>% transform(tfidf)








