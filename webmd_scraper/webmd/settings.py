BOT_NAME = 'webmd'

SPIDER_MODULES = ['webmd.spiders']
NEWSPIDER_MODULE = 'webmd.spiders'

#ROBOTSTXT_OBEY = True

# DOWNLOAD_DELAY = 2.5

# CONCURRENT_REQUESTS = 100

ITEM_PIPELINES = {
    #'webmd.pipelines.ValidateItemPipeline': 100, \
    #'webmd.pipelines.WriteItemPipeline': 200,
    'webmd.pipelines.S3Pipeline': 300,
}

AWS_ACCESS_KEY_ID = 'AKIA4OE3BDYDO5V2YBVY'  # replace with your access key
AWS_SECRET_ACCESS_KEY = 'PGFREcVZeCgGY+7uqry6UMXLWOGSrroGFOTJnXFa'  # replace with your secret key
S3_BUCKET_NAME = 'maxim-thesis'  # replace with your S3 bucket name
FILE_KEY = 'data/0_webmd.csv'