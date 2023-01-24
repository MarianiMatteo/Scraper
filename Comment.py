class Comment:
	def __init__(self, category, text, timestamp):
		self.category = category
		self.text = text
		self.timestamp = timestamp
		self.to_remove = False

	def get_comment(self):
		print('Comment: "', self.text, '" was categorized as: ', self.category, ' at: ', self.timestamp)

	def get_item(self, item):
		return item