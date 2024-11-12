#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

import altair as alt


def fit(data,hj_predictions):
    import jax
    from jax import jit
    import jax.numpy as jnp

    import numpyro
    import numpyro.distributions as dist

    from jax.scipy.special import expit
    from jax import random

    from numpyro.distributions import constraints
    from numpyro.infer import Predictive, SVI, Trace_ELBO

    from diffrax import ODETerm, SaveAt, Euler, diffeqsolve

    eps = 10**-5
    data                   = data.loc[data.season!="offseason"]
    data_by_season         = pd.pivot_table(data = data, index=["season"], columns = ["eweek"], values = ["value"] )

    data_by_season.columns = [y for (x,y) in data_by_season.columns]
    data_by_season         = data_by_season[ np.arange(1,33+1) ]
    data_for_optim         = data_by_season.to_numpy()

    user_input             =  jnp.array(hj_predictions).reshape(1,33)
    empty_input            =  jnp.full_like(user_input, jnp.nan)
    training_data          =  np.vstack([data_for_optim, user_input, empty_input ])

    training_data = training_data+0.5
    
    def ode(beta,center,spread,gamma1,gamma2,theta,eta,S0,E0,I0,R0,H0,h0,ntimes):
        initial_state = jnp.array([S0.reshape(1,), E0.reshape(1,),I0.reshape(1,), H0.reshape(1,), R0.reshape(1,), h0.reshape(1,)])

        def R0_time_dep(t,R0,center,spread):
            return R0*(1+(jax.scipy.stats.norm.pdf(t,center,spread) / jax.scipy.stats.norm.pdf(center,center,spread)))
 
        def sir_model(t, y, args):
            S, E, I, H, R, C = y
            R0,center,spread, gamma1,gamma2,theta,eta = args

            beta = R0_time_dep(t,R0,center,spread)

            dSdt = -beta * S * I 
            dEdt =  beta * S * I    - theta*E 
            dIdt = theta*E          - gamma1 * I
            dHdt = eta*gamma1*I     - gamma2*H
            dRdt = (1-eta)*gamma1*I + gamma2*H
            dCdt = eta*gamma1*I
            return jnp.array([dSdt, dEdt, dIdt, dHdt, dRdt, dCdt])

        ode_term = ODETerm(sir_model)

        # Set up the solver with parameters
        solver = Euler()  

        # Set up the time span and save settings
        t0, t1  = -1, ntimes  
        dt      = 1./7        
        save_at = SaveAt(ts=jnp.arange(t0, t1, 1))

        # Run the solver
        solution = diffeqsolve(
            ode_term,
            solver,
            t0     = t0,
            t1     = t1,
            dt0    = dt,
            y0     = initial_state,
            args   = (beta, center,spread, gamma1,gamma2,theta,eta),
            saveat = save_at)
        C = solution.ys[:,-1,0]
        inc           = jnp.diff(C)
        return inc

    
    def model(data, final_size):
        season_plate = numpyro.plate("season", dim=-2,size=data.shape[0])
        times_plate  = numpyro.plate("times" , dim=-1,size=data.shape[1])

        nseasons,ntimes = data.shape
        times           = jnp.arange(ntimes)

        #@jit
        # Set up the ODE term for the differential equation
      
        phi           = numpyro.sample("phi", dist.Beta(1,1))

        beta           = numpyro.sample("beta"          , dist.Gamma(2,1))
        gamma1         = numpyro.sample("gamma1"        , dist.Gamma(2,1))
        gamma2         = numpyro.sample("gamma2"        , dist.Gamma(2,1))

        eta           = numpyro.sample("eta"            , dist.Beta(1,1) )
        
        theta_a          = numpyro.sample("theta_a"     , dist.Gamma(2,1) )
        theta_b    = numpyro.sample("theta_b"           , dist.Gamma(2,1) ) 
        
        center        = numpyro.sample("center"         , dist.Uniform(0,33))
        
        spread        = numpyro.sample("spread"         , dist.Uniform(0,33))
        
        sigma         = numpyro.sample("sigma"          ,dist.HalfCauchy(10**2))

        inits         = numpyro.sample("inits", dist.Dirichlet( 1*jnp.array([1,1,1,1,1]))) 

        E0,I0,R0,H0,h0 = inits*0.10
        S0             = jnp.array([0.90])
        
        #--season level
        sigma_rw                =  numpyro.sample("sigma_rw"          , dist.Gamma(2,0.2) )
        increments_season       =  numpyro.sample("increments_season" , dist.Normal(0,1./jnp.sqrt(sigma_rw)).expand([1,32]) )
        #increments_season_begin =  numpyro.sample("increments_season_begin" , dist.Normal(0,1./jnp.sqrt(10*sigma_rw)).expand([1,1]) )

        increments_season_begin =  jnp.array([0.]).reshape(1,1)
        increments_season       =  jnp.hstack([increments_season_begin, increments_season]) 
        
        z_season          =  jnp.cumsum(increments_season,axis=-1)[:,::-1]

        weights_for_rw__strength  = numpyro.sample("weights_for_rw__strength", dist.Gamma(2,2) )

        prec_as                   = numpyro.sample("prec_as", dist.Gamma( 1,1) )
        prec_bs                   = numpyro.sample("prec_bs", dist.Gamma( 1,1 ) )
        with season_plate:

            theta    = numpyro.sample("theta_season", dist.Gamma(theta_a, theta_b))
            
            incs   = jax.vmap( ode, in_axes = (None,None,None, None, None,0,None,None,None,None,None,None,None,None) )( beta,center,spread,gamma1,gamma2,theta,eta,S0,E0,I0,R0,H0,h0, ntimes )
            #incs   = ode( beta,center,spread,gamma1,gamma2,theta,eta,S0,E0,I0,R0,H0,h0,ntimes )

            weights_for_rw   = numpyro.sample("weights_for_rw"  , dist.TruncatedNormal( jax.scipy.special.logit(0.95), weights_for_rw__strength
                                                                                        , low = jax.scipy.special.logit(0.02), high=jax.scipy.special.logit(0.98)   ) )
            weights_for_rw   = jax.scipy.special.expit(weights_for_rw)

            def multiply(carry,array):
                return carry*carry, carry
            weights_for_rw = jax.vmap( lambda weight:  jax.lax.scan( multiply
                                                                     ,init = weight
                                                                     ,xs   = jnp.arange(ntimes) )[-1]  ) (weights_for_rw)
            weights_for_rw = weights_for_rw.reshape(nseasons,ntimes)
            random_walk_sigma           = numpyro.sample("random_walk_sigma", dist.HalfNormal(10))# dist.Gamma( prec_as, prec_bs ) )

            with times_plate:
                increments              = numpyro.sample("increments_for_time" , dist.Normal(0, 1./jnp.sqrt(random_walk_sigma)))
                increments              = increments.at[:,0].set( 0 ) #--this makes sure the sum below end at logit(season)

                increments              = increments*weights_for_rw
                random_walk_adjustment  = jnp.cumsum(increments,axis=-1)[:,::-1]

                N = final_size*100
                curve =  (phi*(N) )*( eps + jax.scipy.special.expit( jax.scipy.special.logit(incs) + z_season + random_walk_adjustment ))

                numpyro.deterministic("curve", curve)
                
                with numpyro.handlers.mask(mask=~jnp.isnan(data)):
                    numpyro.sample("obs", dist.NegativeBinomial2( jnp.clip(curve,10**-5,jnp.inf), sigma), obs = data)


    from  numpyro.infer import MCMC, NUTS, HMC, init_to_value, init_to_median, init_to_sample,init_to_uniform
    rng_key    = jax.random.PRNGKey(20201017)
    final_size = jnp.nanmean(jnp.nansum( data_for_optim, axis=1 )) 

    #@st.cache_data
    @st.cache_resource
    def initialrun():
        #--inital values
        mcmc_init = MCMC(
        NUTS(model
             , dense_mass             = False
             , max_tree_depth         = 2
             , regularize_mass_matrix = True
             , init_strategy          = init_to_value())
        , num_warmup  = 5000
        , num_samples = 10000
        , num_chains  = 1
        , thinning    = 2
        )
        #--MCMC RUN
        mcmc_init.run(rng_key
                  , data = jnp.vstack([data_for_optim+0.5, empty_input, empty_input])
                  , final_size = final_size
                 )
        samples__init = mcmc_init.get_samples()

        print(mcmc_init.print_summary())
        
        init_params = {name: np.mean(value,0) for name, value in samples__init.items()}
        return init_params
    #init_params = initialrun()

    #--now with users
    mcmc = MCMC(
    NUTS(model
         , dense_mass             = False
         , max_tree_depth         = 2
         , regularize_mass_matrix = True
         , init_strategy          = init_to_value())# init_to_value(values = init_params))
    , num_warmup  = 1000
    , num_samples = 3000
    , num_chains  = 1
    , thinning    = 2
    )

    #--MCMC RUN
    mcmc.run(rng_key
             , data = training_data
             , final_size = final_size
             )
    samples =  mcmc.get_samples()

    print(mcmc.print_summary())
    
    predictive  = Predictive(model, posterior_samples=samples)
    predictions = predictive(rng_key
                             , data =  training_data
                             , final_size = final_size
                             )
    
    print(predictions["curve"].shape)
    _2p5,_10,_25,_50,_75,_90,_97p5 = np.percentile( predictions["curve"][:,-1,:] , [2.5,10,25,50,75,90,97.5], axis=0 )
    print(_50)
    forecast_data =  pd.DataFrame({ "eweek":np.arange(1,33+1), "_2p5":_2p5, "_10":_10,"_25":_25, "_75":_75,"_90":_90, "_97p5":_97p5, "_50":_50 } )


    _2p5,_10,_25,_50,_75,_90,_97p5 = np.percentile( predictions["curve"][:,-2,:] , [2.5,10,25,50,75,90,97.5], axis=0 )
    print(_50)
    forecast_data__user =  pd.DataFrame({ "eweek":np.arange(1,33+1), "_2p5":_2p5, "_10":_10,"_25":_25, "_75":_75,"_90":_90, "_97p5":_97p5, "_50":_50 } )
   
    return forecast_data, forecast_data__user

if __name__ == "__main__":
    

    @st.cache_data
    def load_data():
        from datetime import datetime, timedelta
        from epiweeks import Week
        
        hosps = pd.read_csv("https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/main/target-data/target-hospital-admissions.csv")
        pa_hosps = hosps.loc[hosps.location=="42"]

        def add_season(x):
            dat = Week.fromdate(datetime.strptime(x.date,"%Y-%m-%d"))
            y,w = dat.year, dat.week
            
            if 40<=w<=53:
                return "{:d}/{:d}".format(y,y+1)
            elif 1<=w<=20:
                return "{:d}/{:d}".format(y-1,y)
            else:
                return "offseason"
        pa_hosps["season"] = pa_hosps.apply(add_season,1)
        
        def add_elapsed_week(x):
            dat = Week.fromdate(datetime.strptime(x.date,"%Y-%m-%d"))
            y,w = dat.year, dat.week

            if 40<=w<=53:
                start = Week(y,40)
            elif 1<=w<=20:
                start = Week(y-1,40)
            else:
                return -1
                
            counter=0
            while start<=dat:
                start+=1
                counter+=1
            return counter
        pa_hosps["eweek"] = pa_hosps.apply(add_elapsed_week,1)
        return pa_hosps

    pa_hosps = load_data()
    pa_hosps = pa_hosps.loc[pa_hosps.season!="offseason"]

    from streamlit_vertical_slider import vertical_slider

    line_chart = alt.Chart(pa_hosps).mark_line().encode(
        x=alt.X('eweek', title='Epidemic week'),  # Custom x-axis label
        y=alt.Y('value', title='Incident hospitalizations'),
        color='season',  # Groups lines by 'season' and adds a legend
        tooltip=['eweek', 'value', 'season']  # Tooltips for interactivity
    )

    # Create the scatter chart with the same grouping by 'season'
    scatter_chart = alt.Chart(pa_hosps).mark_point().encode(
        x=alt.X('eweek', title='Epidemic week'),  # Custom x-axis label
        y=alt.Y('value', title='Incident hospitalizations'),
        color='season',  # Same color grouping as line chart
        tooltip=[ alt.Tooltip('eweek', title='Week')
                 ,alt.Tooltip('value', title='Inc. Hosps')
                  ,alt.Tooltip('season', title='Season')]  # Tooltips for interactivity
    )


    intro = st.container()
    
    
    cols = st.columns([1]*33)
    hj_predictions = []

    avg_vals = pa_hosps.groupby(["eweek"]).apply(lambda x: x['value'].mean())
    for n,col in enumerate(cols):
        with col:
            hj_prediction= vertical_slider(key="hj_prediction_{:d}".format(n), 
                                           default_value=avg_vals[n+1] , 
                                           step=1, 
                                           min_value=0, 
                                           max_value=2000,
                                           slider_color= 'green', #optional
                                           track_color='lightgray', #optional
                                           thumb_color = 'red' #optional
                        )
            hj_predictions.append(hj_prediction)

    hj_df = pd.DataFrame({'eweek': list(np.arange(1,33+1,1)), 'prediction':hj_predictions})
    line_chart__hj = alt.Chart(hj_df).mark_line().encode(
        x=alt.X('eweek', title='Epidemic week'),  # Custom x-axis label
        y=alt.Y('prediction', title='Your prediction'),
        tooltip=['eweek', 'prediction']  # Tooltips for interactivity
    )

    place_holder = st.empty()
    
     # Combine the line and scatter plots using layering
    combined_chart = (line_chart + scatter_chart + line_chart__hj).properties(
        title="Incident hospitalizations in PA"
    ).interactive()  # Enables zooming and panning

    # Display the chart in Streamlit
    place_holder.altair_chart(combined_chart, use_container_width=True)    # Display the chart in Streamlit

    # Initialize the session state for the button if not already set
    if "button_clicked" not in st.session_state:
        st.session_state["button_clicked"] = False

    st.button("Submit", on_click = lambda: st.session_state.update(button_clicked=True))

    if st.session_state["button_clicked"]:
        st.session_state["button_clicked"] = False
        
        with st.spinner(text="Blending the data and your prediction (about 1min)..."):
            forecast_data, forecast_data__user = fit(pa_hosps, hj_predictions = hj_predictions)

        #--create plot
        fill_between1 = alt.Chart(forecast_data).mark_area(opacity=0.2).encode(
            x='eweek',
            y='_2p5',
            y2='_97p5'
        )
        fill_between2 = alt.Chart(forecast_data).mark_area(opacity=0.2).encode(
            x='eweek',
            y='_25',
            y2='_75'
        )
        fill_between3 = alt.Chart(forecast_data).mark_area(opacity=0.2).encode(
            x='eweek',
            y='_10',
            y2='_90'
        )
        line_chart__chimeric = alt.Chart(forecast_data).mark_line().encode(
        x=alt.X('eweek', title='Epidemic week'),  # Custom x-axis label
        y=alt.Y('_50', title=''),
        tooltip=['eweek', '_50']  # Tooltips for interactivity
        )

        #--create plot
        fill_between1__U = alt.Chart(forecast_data__user).mark_area(opacity=0.2).encode(
            x='eweek',
            y='_2p5',
            y2='_97p5'
            ,color = alt.value('yellow')
        )
        fill_between2__U = alt.Chart(forecast_data__user).mark_area(opacity=0.2).encode(
            x='eweek',
            y='_25',
            y2='_75'
            ,color = alt.value('yellow')
        )
        fill_between3__U = alt.Chart(forecast_data__user).mark_area(opacity=0.2).encode(
            x='eweek',
            y='_10',
            y2='_90'
            ,color = alt.value('yellow')
        )
        line_chart__U = alt.Chart(forecast_data__user).mark_line().encode(
        x=alt.X('eweek', title='Epidemic week'),  # Custom x-axis label
        y=alt.Y('_50', title=''),
        tooltip=['eweek', '_50']  # Tooltips for interactivity
            ,color = alt.value('yellow')
        )

        combined_chart1 = (line_chart + scatter_chart+fill_between1+fill_between2+fill_between3 + line_chart__hj+ line_chart__chimeric + fill_between1__U + fill_between2__U+ fill_between3__U + line_chart__U ).properties(
            title="Incident hospitalizations in PA"
        ).interactive()  # Enables zooming and panning

        place_holder.altair_chart(combined_chart1, use_container_width=True)    # Display the chart in Streamlit
