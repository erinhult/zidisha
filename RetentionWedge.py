
# coding: utf-8

# In[14]:

import pymysql as mdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime 
import matplotlib.dates as mdates
import matplotlib.cbook as cbook

plt.rcParams['figure.figsize'] =10,5 # plotsize


# In[15]:

db = mdb.connect(user="root", host="localhost", db="zidisha", charset='utf8')
dfloans = pd.read_sql("SELECT * from loanapplic  ", db)

dfbids = pd.read_sql("SELECT * from loanbids  ", db)

dflenders = pd.read_sql("SELECT * from lenders  ", db)



# Retention Wedge diagram shown by Jonathan Hsu

# In[15]:




# In[16]:

ddates=dfloans[['expires','applydate','AcceptDate','ActiveDate','RepaidDate','auth_date','active','reqdamt']]
ddates.columns=['expires','applydate','AcceptDate','ActiveDate','RepaidDate','auth_date','loanstat','reqdamt']

#add loan status to bid database
dfloansS=dfloans[['loanid','active','borrowerid']]
dfloansS.columns = ['loanid','loanstat','borrowerid']
dfbids=pd.merge(dfbids, dfloansS, left_on='loanid',right_on='loanid',how='left')


# In[19]:

startT=(datetime.datetime(2010,1,1))


# In[19]:




# In[20]:

#How many lenders joined in each month time?
dfbids['biddate']=dfbids['biddate'].apply(float)
grpLI=dfbids[['lenderid','biddate']].groupby('lenderid')
bidbyLender=grpLI.agg({'biddate':np.min})
bidbyLender=bidbyLender.reset_index()
bidbyLender.columns=['lenderid','firstbid']
grpLImax=dfbids[['lenderid','biddate']].groupby('lenderid')
tempM=grpLImax.agg({'biddate':np.max})
tempM.columns=['lastbid']
bidbyLender=pd.merge(bidbyLender,tempM,left_on='lenderid',right_index=1,how='left')


# In[21]:

curTime=time.mktime(datetime.datetime.utcnow().timetuple())-(86400*365*3/12)


# In[22]:

monDat=pd.DataFrame(data=[datetime.datetime(2010 + (x-x%12)/12,x%12+1,1) for x in range(0,56)])
monDat.columns=['startMonth']
monDat['endMonth']=[datetime.datetime(2010 + (x-x%12)/12,x%12+1,1) for x in range(1,57)]
monDat['startME']=[float(time.mktime(x.timetuple())) for x in monDat['startMonth']]
monDat['endME']=[float(time.mktime(x.timetuple())) for x in monDat['endMonth']]
monDat['numRecruit']=[sum((bidbyLender['firstbid']>x)                          & (bidbyLender['firstbid']<y)) for x,y                           in zip(monDat['startME'],monDat['endME'])]
zip(monDat['startME'],monDat['endME'])[1]
monDat['numRecruit'][monDat['numRecruit']==0]=np.nan


# In[23]:

retention=np.zeros([len(monDat),len(monDat)])*NaN
bidbyLender=bidbyLender[bidbyLender['lenderid']>0]
for j in range(len(monDat)):

    #pull just the bids when biddate is between specified start and end date
    # where j steps through the months
    dbtemp=dfbids[(dfbids['biddate']>(monDat['startME'][j]))&                (dfbids['biddate']<(monDat['endME'][j]))]

    #group those bids by lender 
    gtemp=dbtemp[['lenderid','biddate']].groupby('lenderid').size()
    gtemp=gtemp.reset_index()
    # merge those values onto a the list of all lender id's w/ firstbid 
    tempMerge=pd.merge(bidbyLender,gtemp,left_on='lenderid',right_on='lenderid',how='right')
    for i in range(0,len(monDat)):
        if(i<=j):
    #Each element of the array is the fraction of the cohort active in that month
            retention[i,j-i]=(len(tempMerge[(tempMerge['firstbid']>(monDat['startME'][i]))&                                 (tempMerge['firstbid']<(monDat['endME'][i]))])*1.0 )                                    /(monDat['numRecruit'][i]+0.0)
        


# In[24]:


fig, ax = plt.subplots(1)
masked_array = np.ma.array (retention, mask=np.isnan(retention))

p = ax.pcolor(masked_array*100,vmax=100,vmin=0,cmap='jet')
yloc=range(0,len(monDat['startMonth']),3)
yloc=[x+.5 for x in yloc]
ax.set_yticks(yloc)
ydates=monDat['startMonth'].iloc[range(0,len(monDat['startMonth']),3)].tolist()
ydates=[x.strftime('%b %Y') for x in ydates]
ax.set_yticklabels(ydates,rotation=0)
plt.ylabel('Date of first bid',size='large')
plt.xlabel('Months since first bid',size='large')

plt.tick_params(axis="both", which="both", bottom="off", top="off",  
                labelbottom="on", left="on", right="off", \
                labelleft="on",direction='out') 
ax.spines["top"].set_visible(False)  
ax.spines["bottom"].set_visible(False)  

ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

cbar=fig.colorbar(p)
cbar.set_label('% making at least 1 bid',size='large')

fig.savefig('wedge.png',dpi=120)


# The figure below shows the number of recruits in each monthly period

# In[25]:

plt.plot(list(monDat['startMonth']),monDat['numRecruit'])
ylabel('number of recruits')


# The figures below are designed to investigate the potential dip in activity during the month following joining / first bid:

# In[26]:

fig,ax=plt.subplots(2)
Nmax=12
ax[0].pcolor(masked_array[24:-Nmax,0:Nmax]*100,vmax=100,vmin=0,cmap='jet')
ax[1].plot(np.median(masked_array[24:-Nmax,0:Nmax],0))
ylim([.2,.3])


# In[27]:




# In[27]:




# In[ ]:



