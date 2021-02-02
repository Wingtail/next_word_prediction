from nltk.corpus import gutenberg, brown
import os

def _preprocess(raw_text):
    raw_text = (raw_text.replace('\n', ' ').replace('\r', ' ').replace('\ufeff', ' ').replace('_','')).lower()
    return raw_text

def get_gutenberg():
    #Gutenberg corpus
    total_data_count = 0
    for fileid in gutenberg.fileids():
        print("Gutenberg fileid: ", fileid)
        with open("./text_data/"+fileid, "w") as f:
            f.write(gutenberg.raw(fileid))

def get_brown():
    for fileid in brown.fileids():
        print("Brown fileid: ", fileid)
        raw_text = brown.words(fileid)
        raw_text = ' '.join(raw_text)
        with open("./text_data/"+fileid+".txt", "w") as f:
            f.write(raw_text)

def main():
    if not os.path.exists("./text_data/"):
        os.makedirs("./text_data/", exist_ok=True)
    get_gutenberg()
    get_brown()

if __name__ == "__main__":
    main()

