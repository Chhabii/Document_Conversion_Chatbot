import pymupdf
import re
class ProcessDocs:
    def __init__(self,pdf_path):
        self.pdf_path = pdf_path
    
    # text extraction from pdf file
    def text_extraction(self):
        #open a pdf
        doc = pymupdf.open(self.pdf_path)
        # create a text output
        # output_text = open('output.txt','wb')
        output_text = ""
        
        for page in doc: #iterate over the pdf pages
            # text = page.get_text().encode("utf8") #get plain text 
            # output_text.write(text) #write text from the page to the output file
            # output_text.write(bytes((12,))) # write page delimeter
        # output_text.close() # stop writing
            output_text+=page.get_text()

        return output_text
    
    # using regex to discard some part of the text
    def text_cleaning(self,text):
        #normalize the whitespace
        text = re.sub(r'\s+',' ',text)
        """
        remove unnecessary characters, don't remove important puncuations like @, #, %, etc.
        because document contains legal contracts or technical specifications

        """
        text = re.sub(r'[^\w\s.,;:]','',text)
        return text.strip()

    def text_chunks(self,text,chunk_size=350):#384 for all_mpnet #chunk size depends upon the max_sequence_length of sentence transformers to avoid truncation.
        """Convert full text into smaller chunks """

        words = text.split()
        # list of chunks of 500 words 
        chunks = [' '.join(words[i:i+chunk_size]) for i in range(0,len(words),chunk_size)]

        return chunks 

    
    def preprocess(self): #define all the preprocess steps
        raw_text = self.text_extraction()
        clean_text = self.text_cleaning(raw_text)
        chunks = self.text_chunks(clean_text)

        return chunks


