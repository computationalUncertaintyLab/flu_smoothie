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
    
    def model(data, final_size):
        season_plate = numpyro.plate("season", dim=-2,size=data.shape[0])
        times_plate  = numpyro.plate("times" , dim=-1,size=data.shape[1])

        nseasons,ntimes = data.shape
        times           = jnp.arange(ntimes)

        #@jit
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

        # Set up the ODE term for the differential equation
        ode_term = ODETerm(sir_model)

        # Set up the solver with parameters
        solver = Euler()  

        # Set up the time span and save settings
        t0, t1  = -1, ntimes  
        dt      = 1./7        
        save_at = SaveAt(ts=jnp.arange(t0, t1, 1))

        def ode(beta,center,spread,gamma1,gamma2,theta,eta,S0,E0,I0,R0,H0,h0):
            initial_state = jnp.array([S0.reshape(1,), E0.reshape(1,),I0.reshape(1,), H0.reshape(1,), R0.reshape(1,), h0.reshape(1,)])
            
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
        
        phi           = numpyro.sample("phi", dist.Beta(1,1))

        beta          = numpyro.sample("beta"           , dist.Beta(1,1))
        beta_sigma    = numpyro.sample("beta_sigma"     , dist.HalfCauchy(100) )

        gamma1        = numpyro.sample("gamma1"         , dist.Gamma(2,1))
        gamma2        = numpyro.sample("gamma2"         , dist.Gamma(2,1))
        
        eta           = numpyro.sample("eta"            , dist.Beta(1,1))
        
        theta         = numpyro.sample("theta"          , dist.Gamma(2,1))
        
        center        = numpyro.sample("center"         , dist.Normal(0,10**3))
        center        = 33*jax.scipy.special.expit(center)
        
        spread        = numpyro.sample("spread"         , dist.Uniform(0,33))
        
        sigma         = numpyro.sample("sigma",dist.HalfCauchy(100.))

        inits         = numpyro.sample("inits", dist.Dirichlet(jnp.array([1,1,1,1,1,1]))) 
        S0,E0,I0,R0,H0,h0 = inits

        #--season level
        sigma_rw          = numpyro.sample("sigma_rw"         , dist.Gamma(2,0.02) )
        increments_season = numpyro.sample("increments_season", dist.Normal(0,1./jnp.sqrt(sigma_rw)).expand([1,33]) )
        z_season          = jnp.cumsum(increments_season,axis=-1)[:,::-1]

        with season_plate:
            beta       = numpyro.sample("beta_season", dist.Beta(beta*(beta_sigma), (1-beta)*(beta_sigma) ) )
            beta       = 10*beta
            
            incs   = jax.vmap( ode, in_axes = (0,None,None, None, None, None,None,None,None,None,None,None,None) )( beta,center,spread,gamma1,gamma2,theta,eta,S0,E0,I0,R0,H0,h0 )

            N = final_size*10
            curve = (phi*(N) )*jax.scipy.special.expit( jax.scipy.special.logit(incs) + z_season  ) + eps
            numpyro.deterministic("curve", curve)

            with times_plate:
                with numpyro.handlers.mask(mask=~jnp.isnan(data)):
                    numpyro.sample("obs", dist.NegativeBinomial2( jnp.clip(curve,10**-5,jnp.inf), sigma), obs = data)

    num_warmup  = 1000 
    num_samples = 4000
    num_chains  = 1

    from  numpyro.infer import MCMC, NUTS, init_to_value, init_to_median, init_to_sample,init_to_uniform

    mcmc = MCMC(
    NUTS(model
         , dense_mass             = False
         , max_tree_depth         = 2
         , regularize_mass_matrix = True
         , init_strategy          = init_to_value())
    , num_warmup  = num_warmup
    , num_samples = num_samples
    , num_chains  = num_chains
    , thinning    = 2
    )
  
    rng_key = jax.random.PRNGKey(20201017)

    final_size = jnp.nanmean(jnp.nansum( data_for_optim, axis=1 )) 

    #--MCMC RUN
    mcmc.run(rng_key
             ,data = training_data
             , final_size = final_size
             )
    samples = mcmc.get_samples()
    mcmc.print_summary()

    predictive  = Predictive(model, posterior_samples=samples)
    predictions = predictive(rng_key
                             , data =  training_data
                             , final_size = final_size
                             )
    _2p5,_10,_25,_50,_75,_90,_97p5 = np.percentile( predictions["curve"][:,-1,:] , [2.5,10,25,50,75,90,97.5], axis=0 )
    forecast_data =  pd.DataFrame({ "eweek":np.arange(1,33+1), "_2p5":_2p5, "_10":_10,"_25":_25, "_75":_75,"_90":_90, "_97p5":_97p5, "_50":_50 } )

    return forecast_data

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

    #col2 = st.columns([5])
    
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
        
        with st.spinner(text="Blending the data and your prediction..."):
            forecast_data = fit(pa_hosps, hj_predictions = hj_predictions)

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
        y=alt.Y('_50', title='Your prediction'),
        tooltip=['eweek', '_50']  # Tooltips for interactivity
        )

        combined_chart1 = (line_chart + scatter_chart+fill_between1+fill_between2+fill_between3 + line_chart__chimeric).properties(
            title="Incident hospitalizations in PA"
        ).interactive()  # Enables zooming and panning

        place_holder.altair_chart(combined_chart1, use_container_width=True)    # Display the chart in Streamlit
