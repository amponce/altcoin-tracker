class Post:
    def __init__(self, title, score, author, num_comments, url, id=None, text=''):
        self.title = title
        self.score = score
        self.author = author
        self.num_comments = num_comments
        self.url = url
        self.id = id
        self.text = text


class Comment:
    def __init__(self, author, score, body, depth=0, id=None):
        self.id = id
        self.author = author if author else '[deleted]'
        self.score = score
        self.body = body
        self.depth = depth
        self.children = []
        self.collapsed = False
        self.has_more_replies = False
        self.is_root = depth == 0
