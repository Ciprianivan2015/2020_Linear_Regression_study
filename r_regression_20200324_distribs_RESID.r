library('dplyr')
library('ggplot2')

rm()
my_x = seq( from = 500, to = 24500 , by = 1)
alfa = 40
beta = 0.65
my_y = alfa + beta * my_x

df = data.frame( x = my_x, y = my_y, type = 'Linear_Equation' )  
n = nrow( df )
ampl = max( my_y ) - min( my_y )

resid_1 = runif( n = n, min = 0, max = 1 ) * ampl /4
resid_2 = rnorm( n = n, mean = 0, sd = 1 ) * ampl /4
resid_3 = rt( n = n, df = n - 2 ) * ampl /4
resid_4 = ( rbinom( n = n, size = 10, prob = .75 ) - 7.5 ) * ampl /4
resid_5 = ( rchisq( n = n, df = 50, ncp = 5 ) - 200) * ampl / 30 + 12000

df = rbind( df, data.frame( x = my_x, y = my_y + resid_1, type = 'Resid = Uniform'))
df = rbind( df, data.frame( x = my_x, y = my_y + resid_2, type = 'Resid = Normal'))
df = rbind( df, data.frame( x = my_x, y = my_y + resid_3, type = 'Resid = t-Student'))
df = rbind( df, data.frame( x = my_x, y = my_y + resid_4, type = 'Resid = Binomial'))
df = rbind( df, data.frame( x = my_x, y = my_y + resid_5, type = 'Resid = Chi_Square'))


# ............ visualise datasets ........................
ggplot( data = df, aes(x, y, colour = type )) + 
          geom_point( size = 1) + facet_grid( facets = type ~ . )

df_regr = data.frame( x_mean = 0, y_mean = 0, Sxx = 0, Syy = 0, Sxy = 0,  
                      beta_hat = beta, alfa_hat = alfa, SSR = 0, R2 = 1, r =1,
                      type = 'Linear_Equation'
                      )
df_regr_resid = data.frame ( resid = 0, type = 'ZERO' )
df_regr_predict_interv = data.frame( x_pi = 0, y_low = 0, y_high = 0, type = 'ZERO' ) 
x_pi = seq( from = min( my_x ) * 0.25, to = max( my_x )*1.5, by = 1 )
df_types = df %>% filter( type != 'Linear_Equation' ) %>%
                 group_by( type ) %>%
                 summarise( nr = n() ) %>% select( type )

for( iter in df_types$type){
        print( iter )
        df_temp = df %>% filter( type == iter )
        x_mean = mean( df_temp$x )
        y_mean = mean( df_temp$y )
        
        Sxx = sum( ( df_temp$x - x_mean )^2 )
        Syy = sum( ( df_temp$y - y_mean )^2 )
        Sxy = sum( ( df_temp$y - y_mean ) * ( df_temp$x - x_mean ) )
        beta_hat = Sxy / Sxx
        alfa_hat = y_mean - beta_hat * x_mean
        SSR = sum( ( df_temp$y - (alfa_hat + beta_hat * df_temp$x) )^2 )
        R2 = ( Syy - SSR )/Syy
        r = sign( beta_hat ) * sqrt( R2 )
        df_regr = rbind( df_regr, data.frame( x_mean, y_mean, Sxx, Syy, Sxy,  
                                              beta_hat, alfa_hat, SSR, R2, r,
                                              type = iter ) 
                         )
        df_regr_resid = rbind( df_regr_resid, 
                               data.frame( resid = df_temp$y - ( alfa_hat + beta_hat * df_temp$x),
                                           type = iter
                                        ) 
                               )
        # ......... Prediction interval ................
        pi_confid = 99
        t_a = qt( (100-pi_confid) / 100 / 2 , df = n - 2 )
        my_sqrt = sqrt( ( 1 + 1 / n + ( x_pi - x_mean )^2 /Sxx) *SSR / (n-2)  )
        y_low  = alfa_hat + beta_hat * x_pi + t_a * my_sqrt
        y_high = alfa_hat + beta_hat * x_pi - t_a * my_sqrt
        df_regr_predict_interv = rbind( df_regr_predict_interv, x_pi, 
                                        y_low, y_high, type = iter 
                                        )
}      # ..... end of FOR ..............

df_regr_resid = df_regr_resid %>% filter( type != 'ZERO' & type != 'Resid = Binomial')
ggplot( data = df_regr_resid, aes( x = resid, fill = type ) ) +
        geom_histogram( binwidth = n / 50, colour = 'darkgray' ) +
        facet_wrap( facets = ~ type, ncol = 2, nrow = 2 )

df_qnt_regr = df_regr_resid %>% 
                        group_by( type ) %>% 
                        summarise( qt_50 = quantile( resid , probs = 0.50 ) ,
                                   qt_95 = quantile( resid , probs = 0.95 ), 
                                   qt_99 = quantile( resid , probs = 0.99 ) 
                                   )


df = inner_join( x = df, y = df_regr, by = 'type' )
ggplot( data = df, aes( x,y, colour = type ) ) + 
        geom_point( size = 1 ) +
        geom_abline( aes( slope = beta_hat, intercept = alfa_hat ),
                     linetype = 'dashed', colour = 'red', size = 1 ) +
        geom_text( aes( x = 4000, y = -2000,
                        label = paste0( 'R2 = ', round(R2,3), ' beta = ', 
                                        round( beta_hat, 3 ) ) ) ,
                        colour = 'darkred'
                   ) +
        ggtitle( label = 'Comparison of LINEAR REGRESSION performance on several types of residuals' )
                   


