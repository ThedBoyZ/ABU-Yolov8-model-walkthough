import simple_image_download.simple_image_download as simp

my_downloader = simp.Downloader()
# my_downloader.search_urls('Landsapes',limit=10, verbose=True)
# Get List of Saved URLs in cache
print(my_downloader.get_urls())
# Prints the Whole Cache
print(my_downloader.cached_urls)

# Download + search file
my_downloader.directory = 'Mouse Computer/'
my_downloader.download('mouse', limit=50)
# Now donwload all the Searched picture
my_downloader.download(download_cache=True)
my_downloader.flush_cache()

# # Change Direcotory
# my_downloader.directory = 'my_dir/'
# # Change File extension type
# my_downloader.extensions = '.jpg'
# print(my_downloader.extensions)
# my_downloader.download('mouse', limit=50, verbose=True)