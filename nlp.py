class Similarity:
    
    @classmethod
    def wordMatch(cls, t1, t2):
        c = 0
        t = 0
        s = set([word.lower() for word in t2.split(' ')])
        
        for word in t1.split(' '):
            if word.lower() in s:
                c += 1
            t += 1
                
        return c/t