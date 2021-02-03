from nltk.corpus import gutenberg, brown, nps_chat, webtext
import os

def get_gutenberg():
    #Gutenberg corpus
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

def get_web_text():
    for fileid in webtext.fileids():
        print("Webtext fileid: ", fileid)
        raw_text = webtext.words(fileid)
        raw_text = ' '.join(raw_text)
        with open("./text_data/"+fileid+".txt", "w") as f:
            f.write(raw_text)

def get_nps_chat():
    for fileid in nps_chat.fileids():
        print("Npschat fileid: ", fileid)
        raw_text = nps_chat.words(fileid)
        raw_text = ' '.join(raw_text)
        with open("./text_data/"+fileid+".txt", "w") as f:
            f.write(raw_text)

def main():
    if not os.path.exists("./text_data/"):
        os.makedirs("./text_data/", exist_ok=True)
    # get_gutenberg()
    # get_brown()
    get_web_text()
    get_nps_chat()

if __name__ == "__main__":
    main()

