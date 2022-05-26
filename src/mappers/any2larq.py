class Any2Larq:
    def __init__(self) -> None:
        self.translator={}

    def translate(self,original:str)->str:
        if original not in self.translator:
           return None
        
        return self.translator[original]