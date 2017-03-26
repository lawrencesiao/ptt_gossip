# -*- coding: utf-8 -*-

import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
from email.mime.image import MIMEImage 

def notification(self,text='好ㄌ歐'):
	
	fromaddr = 'lawrancesiao@gmail.com'
	toaddr = 'lawrancesiao@gmail.com'
	server = smtplib.SMTP('smtp.gmail.com', 587)
	server.starttls()
	server.login(fromaddr, '')

	server.sendmail(fromaddr, toaddr, text)
	server.quit()