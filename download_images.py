from icrawler.builtin import BingImageCrawler

# list of keywords you want images for
keywords = ["ambulance", "police car", "fire truck"]

for word in keywords:
    # create a separate folder for each keyword
    crawler = BingImageCrawler(storage={'root_dir': f'{word}_images'})
    
    # download up to 100 images for each category
    crawler.crawl(keyword=word, max_num=100)
