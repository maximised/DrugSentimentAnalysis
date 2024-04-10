from scrapy.exceptions import DropItem
from scrapy.exporters import CsvItemExporter

class ValidateItemPipeline(object):

    def process_item(self, item, spider):
        if not all(item.values()):
            raise DropItem('Missing Values!')
        else:
            return item

class WriteItemPipeline(object):

    def __init__(self):
        self.filename = '../data/0_webmd.csv'

    def open_spider(self, spider):
        self.csvfile = open(self.filename, 'wb')
        #self.csvfile = open(self.filename, 'w', newline='', encoding='utf-8')  # Change made here
        self.exporter = CsvItemExporter(self.csvfile)
        self.exporter.start_exporting()

    def close_spider(self, spider):
        self.exporter.finish_exporting()
        self.csvfile.close()

    def process_item(self, item, spider):
        self.exporter.export_item(item)
        return item


import boto3
from io import BytesIO
from scrapy.exporters import JsonLinesItemExporter
'''
class S3Pipeline:

    def open_spider(self, spider):
        self.s3_bucket_name = spider.settings.get('S3_BUCKET_NAME')
        self.s3 = boto3.resource('s3')
        self.items_buffer = BytesIO()

    def close_spider(self, spider):
        self.items_buffer.seek(0)
        print(self.s3_bucket_name)
        self.s3.Bucket(self.s3_bucket_name).put_object(Key='data/0_webmd.json', Body=self.items_buffer)
        self.items_buffer.close()

    def process_item(self, item, spider):
        exporter = JsonLinesItemExporter(self.items_buffer)
        exporter.start_exporting()
        exporter.export_item(item)
        exporter.finish_exporting()
        # Important: Seek to the end of the buffer
        self.items_buffer.seek(0, 2)  # 2 means "relative to the end of file"
        return item
'''

class S3Pipeline:

    def open_spider(self, spider):
        self.s3_bucket_name = spider.settings.get('S3_BUCKET_NAME')
        self.s3 = boto3.resource('s3')
        self.items_buffer = BytesIO()
        self.exporter = JsonLinesItemExporter(self.items_buffer)
        self.exporter.start_exporting()

    def close_spider(self, spider):
        file_key = spider.settings.get('FILE_KEY')
        self.exporter.finish_exporting()
        self.items_buffer.seek(0)
        self.s3.Bucket(self.s3_bucket_name).put_object(Key=file_key, Body=self.items_buffer.getvalue())
        self.items_buffer.close()

    def process_item(self, item, spider):
        self.exporter.export_item(item)
        return item