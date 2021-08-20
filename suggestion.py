from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import textdistance

read_dictionary = pd.read_csv('dictionary/id.csv', delimiter = "\t")

name = pd.DataFrame(list(read_dictionary['katakunci']),columns=["name"])

count_vec = CountVectorizer(ngram_range=(1,5),stop_words=["dinas","yang","dan","atau","di","per","berdasarkan"])

count_data = count_vec.fit_transform(name['name'])

list_word = pd.DataFrame(count_data.toarray(),columns=count_vec.get_feature_names())

list_all_word = list(list_word.columns)


###Model autocorrect
def my_autocorrect():
    input_word = input("masukan kata : ")

    similarities = [1-(textdistance.Jaccard(qval=2).distance(v,input_word)) for v in list_all_word]
    
    # dataframe word
    df = pd.DataFrame({'word':list_all_word})
    
    df['Similarity'] = similarities

    output = df.sort_values(['Similarity'], ascending=False).head(1)

    final_autocorrect = str(output['word'].to_string(index=False).strip())

    print("Apakah data yang anda maksud adalah %s" % final_autocorrect)

my_autocorrect()