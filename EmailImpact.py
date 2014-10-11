
# coding: utf-8

# In[2]:

import pymysql as mdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime 
import matplotlib.dates as mdates
import matplotlib.cbook as cbook

#from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier

plt.rcParams['figure.figsize'] =6,4 # plotsize


# In[942]:

db = mdb.connect(user="root", host="localhost", db="zidisha", charset='utf8')
dfloans = pd.read_sql("SELECT * from loanapplic  ", db)
dfbids = pd.read_sql("SELECT * from loanbids  ", db)
dflenders = pd.read_sql("SELECT * from lenders  ", db)


# Training period data
# Determine # of defaults and total amount lended for each lenderid

# In[4]:

lendemail=dflenders[['userid','last_check_in_email']]
lendemail.columns = ['userid','emaildate']
#print(type(lendemail.ix[10,1]))
lendemail['email']=lendemail['emaildate']>datetime.date(2000,1,1)


# First, we'll consider whether engagement (#s or $s) increases after emails were sent 
# 
# (Then later we'll compare those who received emails to those who didn't)

# In[5]:


#Join bid dates with lender info on lender id
bidsA=pd.merge(dfbids[['lenderid','biddate','givenamount']],lendemail,               left_on='lenderid',right_on='userid')
bidsA['biddate']=bidsA['biddate'].apply(int)


#select only lenders with non-zero email dates
bidsB=bidsA[bidsA['emaildate']>datetime.date(2000,1,1)]

#Add a variable, postE, listing time until email date in months
# Positive when email is after biddate, so months BEFORE email
bidsB['preE']=([time.mktime(x.timetuple()) for x in bidsB['emaildate']]                 - bidsB['biddate'].apply(int))/(86400*365*1/12)

NM=2 #number of months before / after to consider

# list true for each bid that is in the before / after window
bidsB['npE']=(bidsB['preE']<NM) & (bidsB['preE']>0)
bidsB['naE']=(bidsB['preE']>-NM) & (bidsB['preE']<0)
bidsB['amtpreE']=bidsB['npE']*bidsB['givenamount']
bidsB['amtaftE']=bidsB['naE']*bidsB['givenamount']
# sum values in table aggregated to lender level
grpBid=bidsB.groupby('lenderid')
LendTab=grpBid.agg({'npE':np.sum,
                   'naE':np.sum,
                   'amtpreE':np.sum,
                   'amtaftE':np.sum,
                   'biddate':np.min,
                   'emaildate':np.min,
                   'preE': np.max})
LendTab=LendTab.reset_index()


# In[946]:

LendTab['dif']=LendTab['naE']-LendTab['npE']
LendTab['difdol']=(LendTab['amtaftE']-LendTab['amtpreE'])/NM


# In[947]:

# Personal Emails were sent to all inactive users (joining before May 2014)
# Does email make a difference? Ie, do more lend (or lend more money) 
# in the months following the email?

# Pull only users joining prior to date listed

olderUsers=LendTab[LendTab['biddate']<time.mktime(datetime.datetime(2014,2,1)                                          .timetuple()) ]
nonzer=olderUsers[olderUsers['difdol']!=0]


# In[948]:

print('Number of older, inactive lenders receiving personal email: %i'% (len(olderUsers)))
print('Number of lenders with non-zero lending after email: %i' % (len(nonzer)))
print('Mean difference in monthly lending (post-pre): $%f' %       (sum(olderUsers['difdol'])/len(olderUsers['difdol'])))
print('Mean difference of non-zero values: $%f' % (sum(nonzer['difdol'])/len(nonzer['difdol'])))


# In[949]:

fig, ax = plt.subplots()
plt.hist(list(olderUsers['difdol']),40);
ylabel('number of lenders')
xlabel('difference in lending');


# The figure above shows the increase in lending amongst inactive lenders who received a personal email from Zidisha staff. Emails were sent in spring of 2014, and comparison was the difference between the mean of two months prior and two months post-email. Most commonly, there was no difference, as users were inactive and remained inactive. A few lenders had made a loan in the two months prior and did not make a loan in the post period (negative values), but a larger number had increased lending in the post period. The mean effect was an increase in lending of $1.19/month, so not a whole lot. 
# 
# ------------------------------------------------------------------------------------------------

# Now, let's look at the impact of emails to those who joined more recently. [Note, we've skipped over those who joined between 2/1 and 5/1 2014. This is in part because for the older users, we wanted to do a before/after comparison, and so did not want users who's before period was less than the full 2 months, and also because the large surge of lenders who joined in April had atypical behavior.
# 
# Here, sometimes the email came pretty quickly after joining, so using before / after email may be challenging. 
# 
# Ideally, we want to compare loans per month ($/mon) for those who got an email, with those joining at the same time who did not. The tricky part is, emails do not always come a consistent length of time after joining, so which period should be used as comparison? Here I calculated the median time between a lender's first bid and receiving the personal email to get an 'effective time until email', that I could use for the control group that did not receive personal emails. 
# 
# 

# In[739]:

grpBidA=bidsA.groupby('lenderid')
aComp=grpBidA.agg({'biddate':np.min,
                     'email':np.min,
                     'emaildate':np.min})
aComp=aComp.reset_index()
newerU=aComp[aComp['biddate']>time.mktime(datetime.datetime(2014,4,25)                                          .timetuple()) ]
newerU['join2email']=([np.nan if pd.isnull(x) else time.mktime(x.timetuple())                        for x in                        newerU['emaildate']]-                        newerU['biddate'])/(86400*365*1/12)
medLag=np.median(newerU['join2email'][newerU['email']==True])
#Set an effective mail date for those in control group
newerU['efEmail']=([(y+medLag*(86400*365*1/12)) if pd.isnull(x) else time.mktime(x.timetuple())                     for x,y in                        zip(newerU['emaildate'],newerU['biddate'])])
newerU.rename(columns={'biddate':'joindate'}, inplace=True)

#recombine at bid level to calculate bids since email date
rebid=pd.merge(newerU[['lenderid','efEmail','joindate','email']],bidsA[['lenderid','biddate',               'givenamount']],
               left_on='lenderid',right_on='lenderid',how='left')

#Calculate money lent in each of the 1,2,3,4th months following email receipt

rebid['mon1lend']=rebid['givenamount']*((rebid['biddate']>rebid['efEmail'])                                        &(rebid['biddate']<(rebid['efEmail']+
                                    (1.*86400*365*1/12))))
rebid['mon2lend']=rebid['givenamount']*((rebid['biddate']>(rebid['efEmail']                                         +(86400*365*1/12)))
                                        &(rebid['biddate']<(rebid['efEmail']+
                                    (2*86400*365*1/12))))
rebid['mon3lend']=rebid['givenamount']*((rebid['biddate']>(rebid['efEmail']                                         +(86400*365*2/12)))
                                        &(rebid['biddate']<(rebid['efEmail']+
                                    (86400*365*3/12))))
rebid['mon4lend']=rebid['givenamount']*((rebid['biddate']>(rebid['efEmail']                                         +(86400*365*3/12)))
                                        &(rebid['biddate']<(rebid['efEmail']+
                                    (86400*365*4/12))))

#total amount given during the half-month before email was received
rebid['prelend']=rebid['givenamount']*((rebid['biddate']>(rebid['efEmail']                                         -(86400*365*.5/12)))
                                        &(rebid['biddate']<(rebid['efEmail']-86400)))
rebid['prelendA']=rebid['givenamount']*((rebid['biddate']>rebid['joindate'])
                                        &(rebid['biddate']<(rebid['efEmail']-86400)))\
                                    /((rebid['efEmail']-rebid['joindate'])/(86400*365/12))

newComp=rebid.groupby('lenderid').agg({'mon1lend':np.sum,
                   'mon2lend':np.sum,
                   'mon3lend':np.sum,
                   'mon4lend':np.sum,
                   'prelend':np.sum,
                   'prelendA':np.sum,
                   'email':np.min,
                   'efEmail':np.min,
                   'joindate':np.min})


# In[740]:

maxdate=max(bidsA['biddate'])
# if the number of months since the email is less than, the post period set amt given to NaN 
newComp['mon4lend'][(newComp['efEmail']+(86400*365*4/12))>maxdate]=np.nan
newComp['mon3lend'][(newComp['efEmail']+(86400*365*3/12))>maxdate]=np.nan
newComp['mon2lend'][(newComp['efEmail']+(86400*365*2/12))>maxdate]=np.nan
newComp['mon1lend'][(newComp['efEmail']+(86400*365*1/12))>maxdate]=np.nan


# In[741]:

#Add columns to show activity in these periods
newComp['preact']=[int(x>0) if np.isfinite(x) else np.nan                    for x in newComp['prelend']]
newComp['preactA']=[int(x>0) if np.isfinite(x) else np.nan                    for x in newComp['prelendA']]
newComp['mon1act']=[int(x>0) if np.isfinite(x) else np.nan                    for x in newComp['mon1lend']]
newComp['mon2act']=[int(x>0) if np.isfinite(x) else np.nan                    for x in newComp['mon2lend']]
newComp['mon3act']=[int(x>0) if np.isfinite(x) else np.nan                    for x in newComp['mon3lend']]
newComp['postSum']=[nanmean([x,y,z]) for x,y,z in                            zip(newComp['mon1lend'],newComp['mon2lend'],                                newComp['mon3lend'])]


# In[950]:

figure(num=None, figsize=(9, 4))
a=subplot(1,2,1)
lendingbymonth=newComp[['prelendA','mon1lend','mon2lend','mon3lend','email',]]    .groupby('email').mean()
plt.bar([1,2,3,4],lendingbymonth.iloc[0,:],width=.3)
plt.bar([1.3,2.3,3.3,4.3],lendingbymonth.iloc[1,:],width=.3,color='red')
plt.xticks([1.3,2.3,3.3,4.3], ['monthly average prior','1st month post',                               '2nd month post',             '3rd month post','4th month post'], rotation='vertical');
a.set_ylabel('average $ loaned/lender')

b=subplot(1,2,2)
activitybymonth=newComp[['preactA','mon1act','mon2act','mon3act','email',]]    .groupby('email').mean()
plt.bar([1,2,3,4],activitybymonth.iloc[0,:],width=.3)
plt.bar([1.3,2.3,3.3,4.3],activitybymonth.iloc[1,:],width=.3,color='red')
plt.xticks([1.3,2.3,3.3,4.3], ['active btwn bid1 & email','1st month post',                               '2nd month post',             '3rd month post','4th month post'], rotation='vertical');
b.set_ylabel('fraction active')


# The graph above compares lending and activity for control (blue) and treatment (red) groups, using all of the data in that time period. The main issue here is that lending levels were quite different before the emails were sent, making it difficult to interpret the lending levels after the email was sent. 

# In[951]:

lendingbymonth


# In[952]:

activitybymonth


# Based on the tables & graph above, there appears to be a big difference in the amount loaned in the 0.5 months prior to receiving the emails ('hmpre'), between the treatment and control groups. Specifically, there are more higher activity users in the treatment group. 

# We can select only users below some threshold from the two groups in order to be able to compare activity in the post-email period. 

# In[953]:

# Select only users below a threshhold
# Here the 'threshhold' is to only include users that were inactive
#    in the 0.5 month period before the email
testComp=newComp[newComp['prelendA']<40]
#testComp=newComp


# The table above compares the mean amount given in the 0.5 months before the email was received between treatment and control groups, provided only users with 0 loans in the 'pre' period. As shown below, this still leaves 250+ participants in each group (although we'll see later only 100+ have data for 1+ months in the post period).

# In[954]:

testComp[['prelend','email']].groupby('email').count()


# In[955]:

newComp[['prelend','email']].groupby('email').count()


# The table below shows how many 

# In[956]:

testComp[['preact','mon1act','mon2act','mon3act','postSum','email']].groupby('email').count()


# In[956]:




# Implement t-test for impact of Emails on retention, lending for new users

# In[957]:

#We can run a t-test to show if 
from scipy.stats import ttest_ind

def emailTtest(dframe, depvar):
    # Run t-test comparing those who received email vs those who didn't
    # Select rows of treatment and control groups that have values for the 
    #    dependent variable of interest 
    # Note: Scipy runs 2-tailed t-test, so for 1-tailed pval, divide by 2
    Treatframe = dframe[(dframe['email']==True) & (dframe[depvar]>-1)]
    Contframe = dframe[(dframe['email']==False) & (dframe[depvar]>-1)]
    # outputs are t-statistic and p-value
    return  ttest_ind(Treatframe[depvar], Contframe[depvar])



# In[958]:

#Dependent variable of interest:
output='postSum'
(tstat,pval)=emailTtest(testComp,output)
print tstat,pval


# In[959]:

#Histogram distributions in the whole data set for treatment and control group
a=subplot(1,2,1)
plt.hist(list(newComp['postSum'][(newComp['email']==True) &                                 (newComp['postSum']>-1)]),40)
a=subplot(1,2,2)
plt.hist(list(newComp['postSum'][(newComp['email']==False) &                                 (newComp['postSum']>-1)]),40);


# Tables below show the mean for test and control groups for activity and lending.

# In[960]:

testComp[['preactA','mon1act','mon2act','mon3act','email']].        groupby('email').mean()


# In[961]:

testComp[['prelendA','mon1lend','mon2lend','mon3lend','postSum','email']].        groupby('email').mean()


# In[962]:

a=subplot(1,2,1)
lendingbymonth=testComp[['prelendA','mon1lend','mon2lend','mon3lend','email',]]    .groupby('email').mean()
plt.bar([1,2,3,4],lendingbymonth.iloc[0,:],width=.3)
plt.bar([1.3,2.3,3.3,4.3],lendingbymonth.iloc[1,:],width=.3,color='red')
plt.xticks([1.3,2.3,3.3,4.3], ['monthly average prior','1st month post',                               '2nd month post',             '3rd month post','4th month post'], rotation='vertical');
a.set_ylabel('average $ loaned/lender')

b=subplot(1,2,2)
activitybymonth=testComp[['preactA','mon1act','mon2act','mon3act','email',]]    .groupby('email').mean()
plt.bar([1,2,3,4],activitybymonth.iloc[0,:],width=.3)
plt.bar([1.3,2.3,3.3,4.3],activitybymonth.iloc[1,:],width=.3,color='red')
plt.xticks([1.3,2.3,3.3,4.3], ['active btwn bid1 & email','1st month post',                               '1st month post',                               '2nd month post',             '3rd month post','4th month post'], rotation='vertical');
b.set_ylabel('fraction active')


# This figure shows mean lending & fraction active by month for case when we've restricted analysis to those with lending of $40/month or less in the period between the first bid and receiving a personal email from Zidisha staff for the treatment group (red) and the control group (blue).   
# 
# Here we see that activity does not appear to be strongly affected by receipt of the email.  
# 
# The mean lending does appear to increase, however, after receiving the email. According to the ttest conducted above, the mean monthly lending was about $11 after receiving a personal email, compared to about $3 monthly lending for those who did not receive the email. This difference in means has a one-tailed p-value of 0.11. I.e., it is more likely than not that those that received a personal email did lend more than those that did not, following receipt of the email, however there is an 11% chance that a difference of this size or greater could have occured through random chance. 

# In[ ]:



