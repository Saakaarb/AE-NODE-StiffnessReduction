import numpy as np
import scipy
from matplotlib import pyplot as plt
import time
from pathlib import Path
import shutil
import multiprocessing as mp
from itertools import product
import os

class two_eq_model():

    def __init__(self):

        
        # declare enthalpies of formation
        self.formation_enthalpies= {'CO2':-393.5*1000,'CO':-110.525*1000,'CH4':-74.81*1000,'O2':0,'H2O':-241.818*1000,'H2':0} # J/mol 

        #J/kgK
        self.Cp={'CO2':0.98*1000,'CO':0.98*1000,'CH4':1.005*1000,'O2':0.980*1000,'H2O':0.98*1000,'H2':0.98*1000,'N2':1.2*1000}        

        # molar masses (kg)
        self.molar_masses={'CO2':44/1000,'CO':28/1000,'CH4':16/1000,'O2':32/1000,'H2O':18/1000,'H2':2/1000,'N2':28/1000}


        self.state_map={0:'O2',1:'CO2',2:'CH4',3:'H2O',4:'H2',5:'N2'}
        self.inverse_state_map={'O2':0,'CO2':1,'CH4':2,'H2O':3,'H2':4,'N2':5}
        #eq1
        
        self.A_1=5.16E13
        #self.A_1=5.16E10
        self.Ea_1=1.3E5
        self.rxn_orders_1={'CH4':1.0,'O2':2.0}
        self.rxn_stoich_coeffs_1={'CH4':1.0,'O2':2.0}
        

        #eq2
        self.A_2=1.08E12
        #self.A_2=1.08E9
        self.Ea_2=1.25E5
        self.rxn_orders_2={'H2':1.0,'O2':1.0}
        self.rxn_stoich_coeffs_2={'H2':1.0,'O2':0.5}

    # state variable mass fractions: [O2,CO2,CH4,H2O,H2,N2,T]
    # state map: [0:O2:CO2,2:CH4,3:H2O,4:H2,5:N2]
    def rate_ODE(self,t,state):
        # constants for the simulation
        R=8.314#287
        pres=101325

        Temp=state[-1]
        
        K1=self.A_1*np.exp(-self.Ea_1/(R*Temp))
        K2=self.A_2*np.exp(-self.Ea_2/(R*Temp))
        

        inv_Meff=0
        for i_state in range(len(state)-1):

            inv_Meff+=state[i_state]/self.molar_masses[self.state_map[i_state]]

        Rm=8.314*inv_Meff

        

        # density
        rho=pres/(Temp*Rm)

        molar_concs=np.zeros([len(state)-1])
        #calculate molar concentrations
        for i_state in range(len(state)-1):
            
            molar_concs[i_state]=state[i_state]*rho/self.molar_masses[self.state_map[i_state]]

        
        #compute rates of reaction
        r1=K1*(molar_concs[self.inverse_state_map['CH4']]**self.rxn_orders_1['CH4'])*(molar_concs[self.inverse_state_map['O2']]**self.rxn_orders_1['O2'])

        r2=K2*molar_concs[self.inverse_state_map['H2']]**self.rxn_orders_2['H2']*molar_concs[self.inverse_state_map['O2']]**self.rxn_orders_2['O2']
        #compute dT_dt

        Q_dot=(-r1*(self.formation_enthalpies['CO2']+2*self.formation_enthalpies['H2O']-self.formation_enthalpies['CH4']) -r2*(self.formation_enthalpies['H2O']))*1

        #compute weighted Cp

        Cp=0
        
                
        for i_state in range(len(state)-1):

            Cp+=self.Cp[self.state_map[i_state]]*state[i_state] #/molar_masses[state_map[i_state]]

        #compute total mass
        
        m_total=rho

        dT_dt=Q_dot/(Cp*m_total)


        #compute rate of change of species
        dO2_dt=(-2*r1-0.5*r2)*self.molar_masses['O2']/rho
        dCO2_dt=(r1)*self.molar_masses['CO2']/rho
        dCH4_dt=(-r1)*self.molar_masses['CH4']/rho
        dH2O_dt=(2*r1+r2)*self.molar_masses['H2O']/rho
        dH2_dt=(-r2)*self.molar_masses['H2']/rho
        dN2_dt=0#(-2*r1-r2)*self.molar_masses['O2']/rho

        return [dO2_dt,dCO2_dt,dCH4_dt,dH2O_dt,dH2_dt,dN2_dt,dT_dt]


    def solve_ODE(self,init_cond):

        t1=time.time()        
        time_span=[0,5*10**(-8)]
        result=scipy.integrate.solve_ivp(self.rate_ODE,time_span,init_cond,method='LSODA')
        t2=time.time()        

        print("time:",t2-t1)
        return result

def solve_single_case(params):
    """
    Solve a single ODE case with given parameters.
    
    Args:
        params: tuple of (ch4_init, T_init, save_dir)
        
    Returns:
        str: filename of saved data
    """
    ch4_init, T_init, save_dir = params
    
    # Initialize the model
    sys = two_eq_model()
    
    # Set up initial conditions
    # state map: [0:O2, 1:CO2, 2:CH4, 3:H2O, 4:H2, 5:N2, 6:T]
    init_cond = [0.21, 0, ch4_init, 0, 0.05, 1-0.21-ch4_init-0.05, T_init]
    
    try:
        # Solve the ODE
        result = sys.solve_ODE(init_cond)
        
        # Create filename in the format data_{ch4_init}_{T_init}.txt
        filename = f"data_{ch4_init:.3f}_{T_init}.txt"
        filepath = save_dir / filename
        
        # Save data
        savearray = np.concatenate([np.expand_dims(result.t, axis=0), result.y], axis=0)
        np.savetxt(filepath, savearray, delimiter=',')
        
        print(f"Generated: {filename}")
        return filename
        
    except Exception as e:
        print(f"Error solving case ch4={ch4_init}, T={T_init}: {e}")
        return None

if __name__=="__main__":

    # Create save directory
    save_dir = Path('data')
    if save_dir.exists():
        shutil.rmtree(save_dir)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Define parameter ranges
    ch4_range = np.linspace(0.01, 0.21, 20)  # 20 CH4 values
    T_range = np.linspace(900, 2000, 15)     # 15 temperature values
    
    # Generate all parameter combinations (300 total)
    param_combinations = list(product(ch4_range, T_range))
    
    print(f"Generating {len(param_combinations)} ODE solutions...")
    print(f"CH4 range: {ch4_range[0]:.3f} to {ch4_range[-1]:.3f}")
    print(f"Temperature range: {T_range[0]:.0f}K to {T_range[-1]:.0f}K")
    print(f"Save directory: {save_dir.absolute()}")
    
    # Prepare parameters for multiprocessing
    params_list = [(ch4, T, save_dir) for ch4, T in param_combinations]
    
    # Use multiprocessing to solve ODEs in parallel
    num_processes = min(mp.cpu_count(), 8)  # Limit to 8 processes to avoid overwhelming the system
    print(f"Using {num_processes} processes...")
    
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(solve_single_case, params_list)
    
    # Count successful generations
    successful = [r for r in results if r is not None]
    print(f"\nSuccessfully generated {len(successful)} out of {len(param_combinations)} files")
    print(f"Files saved to: {save_dir.absolute()}")
    



