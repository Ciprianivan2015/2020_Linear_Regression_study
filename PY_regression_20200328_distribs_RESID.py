# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import t

# .....  define initial sequences .....
# ..... define a range of numbers: use of num.py...
my_x = np.arange( 100, 500 + 1, 10 )
n = len( my_x )
alfa = 40
beta = .65
my_y = alfa + beta * my_x
#print( my_type ) #print( my_x ) #print( n )

ampl = max( my_y ) - min( my_y )

# ........... Generate Residuals for multiple Stat Distributions ...........
resid_1 = np.random.uniform( 0, 1, n ) * ampl / 4 - 30
resid_2 = np.random.normal( 0, 1, n ) * ampl / 4
resid_3 = np.random.standard_t( n-2, n ) * ampl / 4
resid_4 = ( np.random.binomial( 10, 0.75, n ) - 7.5 ) * ampl / 4
resid_5 = ( np.random.noncentral_chisquare( 50, 5, n ) - 200 ) * ampl / 30 + 1250  

# .........  Build the [x,y] DataFrame ......
#df = pd.DataFrame({'x':my_x,'y':my_y })
my_frames =[ pd.DataFrame({'x':my_x,'y':my_y,'type':['Linear_Equation'] * n }), 
             pd.DataFrame({'x':my_x,'y':my_y + resid_1,'type':['Uniform'] * n }),
             pd.DataFrame({'x':my_x,'y':my_y + resid_2,'type':['Normal'] * n }),
             pd.DataFrame({'x':my_x,'y':my_y + resid_3,'type':['T-Stud'] * n }),
             pd.DataFrame({'x':my_x,'y':my_y + resid_4,'type':['Binomial'] * n }),
             pd.DataFrame({'x':my_x,'y':my_y + resid_5,'type':['ChiSquare'] * n })             
             ]

df = pd.concat( my_frames, keys = ['Linear_Equation','Uniform',
                              'Normal','T-Stud', 'Binomial', 'ChiSquare'] )
#print( df )
#df.plot( 'x', 'y', kind = 'scatter' )
#print( df.loc['Normal'] )

# ...................................................................
#         Plot a FacetGrid to see all 6 sub-plots, by 'type'
# ...................................................................
g = sns.FacetGrid( df, col = 'type' , hue = 'type', col_wrap = 3)
g = (g.map(plt.scatter , 'x','y', marker='.' )
        .add_legend()
    )
axes = g.fig.axes
for ax in axes:
    ax.plot( my_x, my_y  )
    ax.plot( my_x, [0] *n  )
    
# ...................................................................
#         Define the starting 'Regr' and 'Regr_Resid' DataFrame-s
# ...................................................................
df_regr = pd.DataFrame({ 'x_mean':[0],'y_mean':[0], 'Sxx':[0],'Syy':[0],'Sxy':[0],
                        'beta_hat':[beta], 'alfa_hat':[alfa], 'SSR':[0],'R2':[1],'r':[1],
                        'type':['Linear_Equation']}
                      )
df_regr_resid = pd.DataFrame({'resid':[0], 'type':['ZERO']})
x_pi = np.arange( min( my_x ) * 0.10, max( my_x ) * 3.5, 10 )
df_regr_predict_interv = pd.DataFrame({ 'x_pi':x_pi, 'y_low':alfa + x_pi*beta, 
                                       'y_high':alfa + x_pi*beta, 
                                       'type':'Linear_Equation',
                                       'conf_level':99
                                       })


my_types = np.ma.array( df.type.unique(), mask=False)
my_types.mask[ 0 ] = True
my_types = my_types.compressed()

for iter_type in my_types:
    print( 'We arrived here: ',iter_type )
    df_temp = df.loc[ iter_type ]
    x_mean = np.array( df_temp.x ).mean()
    y_mean = np.array( df_temp.y ).mean()
    Sxx = sum( (df_temp.x - x_mean )**2 )
    Syy = sum( (df_temp.y - y_mean )**2 )
    Sxy = sum( (df_temp.y - y_mean )*(df_temp.x - x_mean ))
    beta_hat = Sxy / Sxx
    alfa_hat = y_mean - beta_hat * x_mean
    temp_resid = df_temp.y - ( alfa_hat + beta_hat * df_temp.x )
    SSR = sum( temp_resid ** 2 )
    R2 = (Syy - SSR) / Syy
    r =  np.sign(beta_hat) * np.sqrt( R2 )
    my_frame_temp = [ df_regr,  
                      pd.DataFrame({ 'x_mean':[x_mean],'y_mean':[y_mean], 'Sxx':[Sxx],'Syy':[Syy],
                                    'Sxy':[Sxy],'beta_hat':[beta_hat],'alfa_hat':[alfa_hat],'SSR':[SSR],
                                    'R2':[R2],'r':[r],'type':[iter_type]})
                     ]    
    df_regr = pd.concat( my_frame_temp )
    df_regr_resid = pd.concat([df_regr_resid,
                               pd.DataFrame({ 'resid':temp_resid,'type':iter_type })
                               ])
    
    my_sqrt = np.sqrt( ( 1 + 1 / n + ( x_pi - x_mean )**2 / Sxx )*SSR / ( n - 2 ) ) 
    # ...... prediction interval at 99 ............   
    pi_confid = 99
    t_a = t.ppf( (100 - pi_confid ) / 100 / 2, n - 2 )    
    y_low  = alfa_hat + beta_hat * x_pi + t_a * my_sqrt
    y_high = alfa_hat + beta_hat * x_pi - t_a * my_sqrt    
    
    df_regr_predict_interv = pd.concat([ df_regr_predict_interv,  
                                        pd.DataFrame({ 'x_pi':x_pi,'y_low':y_low,
                                                      'y_high':y_high,
                                                      'type':iter_type,
                                                      'conf_level':99
                                                      })
                                        ])
    # ...... prediction interval at 95 ............   
    pi_confid = 95
    t_a = t.ppf( (100 - pi_confid ) / 100 / 2, n - 2 )    
    y_low  = alfa_hat + beta_hat * x_pi + t_a * my_sqrt
    y_high = alfa_hat + beta_hat * x_pi - t_a * my_sqrt    
    
    df_regr_predict_interv = pd.concat([ df_regr_predict_interv,  
                                        pd.DataFrame({ 'x_pi':x_pi,'y_low':y_low,
                                                      'y_high':y_high,
                                                      'type':iter_type,
                                                      'conf_level':95
                                                      })
                                        ])
    
sns.set_style("darkgrid")

g = sns.FacetGrid( df, col = 'type' , hue = 'type', col_wrap = 3)
g = (g.map(plt.scatter , 'x','y', marker='.' )
        .add_legend()
    )


g = sns.FacetGrid( df_regr_predict_interv, col = 'type' , hue = 'type', col_wrap = 3)
g = (g.map( plt.scatter , 'x_pi','y_low', marker = '.' ) )
g = (g.map( plt.scatter , 'x_pi','y_high',  marker = '.' )
        .add_legend())

axes = g.fig.axes
for ax in axes:
    ax.plot( x_pi, alfa + beta * x_pi , linestyle = '-.' )
    ax.plot( x_pi, [0] *len( x_pi ) , linestyle = ':' )
    ax.plot( [min( my_x ), max( my_x )], [min( my_y ), max( my_y )] )
    
    
    
    
    





