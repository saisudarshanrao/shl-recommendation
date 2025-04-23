# test_driver.py
from selenium import webdriver
from selenium.webdriver.chrome.service import Service

driver = webdriver.Chrome(service=Service('/usr/local/bin/chromedriver'))
driver.get("https://www.google.com")
print(driver.title)
driver.quit()
