# Reaction Engineering Model for a Gas-Phase Fixed-Bed Reactor

## 1. Physical reactor

The system is a **gas-phase catalytic fixed-bed reactor** from Micromeritics.

The experimental setup contains:

- four gas mass-flow controllers (MFCs);
    
- a separate liquid feed/vaporizer system;
    
- a fixed-bed reactor containing a known mass of catalyst;
    
- temperature control;
    
- a back-pressure regulator (BPR);
    
- an online gas chromatograph (GC) at the reactor outlet.
    

For the initial model, only **three gas MFCs** will be used:

- MFC 1: reactant $A$
    
- MFC 2: reactant $B$
    
- MFC 3: inert gas $I$
    

The fourth gas MFC is unused.

The liquid feed/vaporizer is also not used in the initial model.

The feed from each active MFC is assumed to be a pure gas:

$$
y_{A,\mathrm{feed}}=1
$$

for the $A$ MFC,

$$
y_{B,\mathrm{feed}}=1
$$

for the $B$ MFC, and

$$
y_{I,\mathrm{feed}}=1
$$

for the inert MFC.

The maximum flow rate of each gas MFC is approximately:

$$
0\leq \dot V_i \leq 50\ {\rm mL\,min^{-1}}
$$

at the reference conditions used by the MFC.

The reactor can operate up to approximately:

$$
T_{\max}=800^\circ{\rm C}
$$

or:

$$
T_{\max}=1073.15\ {\rm K}
$$

and at pressures up to approximately:

$$
P_{\max}=100\ {\rm bar}.
$$

For the initial simulations, the pressure range can be taken as:

$$
1\leq P\leq100\ {\rm bar}.
$$

All reactants and products are assumed to remain in the gas phase.

---

# 2. Variables defining one reactor experiment

A single simulated experiment is defined by:

$$
\boxed{
\dot V_{A,0},
\dot V_{B,0},
\dot V_{I,0},
W_{\mathrm{cat}},
T,
P
}
$$

where:

- $\dot V_{A,0}$ = inlet flow setting of reactant $A$
    
- $\dot V_{B,0}$ = inlet flow setting of reactant $B$
    
- $\dot V_{I,0}$ = inlet flow setting of inert gas
    
- $W_{\mathrm{cat}}$ = catalyst mass
    
- $T$ = reactor temperature
    
- $P$ = reactor pressure
    

The catalyst mass is an important independent variable because it determines the amount of catalytic material available relative to the gas feed.

Increasing catalyst mass at the same inlet flow generally increases the contact time and therefore increases conversion.

---

# 3. Convert MFC flow rates to molar flow rates

The reactor equations should be solved using molar flow rates rather than volumetric flow rates.

Let:

$$
F_{A,0},\quad F_{B,0},\quad F_{I,0}
$$

represent the inlet molar flow rates in units such as:

$$
{\rm mol\,s^{-1}}.
$$

If an MFC reports a standard volumetric flow rate $\dot V_{\mathrm{std}}$, then:

$$
\boxed{
F_i=
\frac{P_{\mathrm{std}}\dot V_{i,\mathrm{std}}}
{RT_{\mathrm{std}}}
}
$$

where $P_{\mathrm{std}}$ and $T_{\mathrm{std}}$ are the reference conditions used by the MFC.

The actual MFC reference conditions should eventually be obtained from the instrument specifications.

For the initial simulated model, one consistent set of standard conditions can be assumed.

---

# 4. Initial reaction network

A simple reaction network containing both a desired and an undesired reaction can be used.

## Desired reaction

$$
\boxed{
A+B\rightarrow C
}
$$

## Side reaction

$$
\boxed{
A\rightarrow D
}
$$

Thus:

- $A$ and $B$ are reactants;
    
- $C$ is the desired product;
    
- $D$ is an undesired product;
    
- $I$ is an inert gas.
    

This simple network is sufficient to generate meaningful changes in:

- conversion;
    
- selectivity;
    
- product yield;
    
- outlet composition.
    

---

# 5. Why catalyst mass is used as the reactor coordinate

For a heterogeneous catalytic fixed-bed reactor, reaction rate is conveniently expressed per unit mass of catalyst.

For example:

$$
r_i \quad [{\rm mol\,g_{cat}^{-1}\,s^{-1}}]
$$

or:

$$
r_i \quad [{\rm mol\,kg_{cat}^{-1}\,s^{-1}}].
$$

The reactor differential equations can therefore be written as a function of catalyst mass $W$:

$$
W=0
$$

at the reactor inlet and:

$$
W=W_{\mathrm{cat}}
$$

at the reactor outlet.

This formulation is usually called a **packed-bed reactor mole balance**.

The general equation is:

$$
\frac{dF_i}{dW}=\sum_j \nu_{ij}r_j
$$

where:

- $F_i$ is the molar flow rate of species $i$;
    
- $W$ is catalyst mass;
    
- $r_j$ is reaction rate per unit catalyst mass;
    
- $\nu_{ij}$ is the stoichiometric coefficient of species $i$ in reaction $j$.
    

---

# 6. State variables

At any position through the catalyst bed, define the molar flow rates:

$$
F_A,\quad F_B,\quad F_C,\quad F_D,\quad F_I.
$$

The reactor state vector is:

$$
\boxed{
\mathbf F=
[F_A,F_B,F_C,F_D,F_I]
}
$$

The total molar flow rate is:

$$
\boxed{
F_T=
F_A+F_B+F_C+F_D+F_I
}
$$

Importantly, $F_T$ does not necessarily remain constant through the reactor.

For example, Reaction 1 is:

$$
A+B\rightarrow C
$$

so two gas molecules are converted into one gas molecule.

Therefore the total molar flow decreases as this reaction proceeds.

---

# 7. Inlet conditions

At the reactor inlet:

$$
F_A(0)=F_{A,0}
$$

$$
F_B(0)=F_{B,0}
$$

$$
F_I(0)=F_{I,0}
$$

and initially there are no products:

$$
F_C(0)=0
$$

$$
F_D(0)=0.
$$

Therefore:

$$
\boxed{
\mathbf F(0)=[F_{A,0},F_{B,0},0,0,F_{I,0}]
}
$$

The inlet total flow is:

$$
F_{A,0}+F_{B,0}+F_{I,0}.
$$

---

# 8. Gas composition inside the reactor

At any position in the catalyst bed, the mole fraction of component $i$ is:

$$
\boxed{
y_i=\frac{F_i}{F_T}
}
$$

Therefore:

$$
y_A=\frac{F_A}{F_T}
$$

$$
y_B=\frac{F_B}{F_T}
$$

and similarly for the remaining components.

The mole fractions should satisfy:

$$
\boxed{
\sum_i y_i=1
}
$$

at every position in the reactor.

---

# 9. Partial pressures

For an ideal gas mixture:

$$
\boxed{
P_i=y_iP
}
$$

Therefore:

$$
P_A=P\frac{F_A}{F_T}
$$

and:

$$
P_B=P\frac{F_B}{F_T}.
$$

Thus changes in:

- total pressure;
    
- feed composition;
    
- conversion;
    
- inert dilution
    

all automatically affect the reactant partial pressures.

---

# 10. Reaction-rate expressions

For the first model, simple power-law kinetics can be assumed.

For:

$$
A+B\rightarrow C
$$

define:

$$
\boxed{
r_1=
k_1(T)
P_A^\alpha
P_B^\beta
}
$$

For:

$$
A\rightarrow D
$$

define:

$$
\boxed{
r_2=
k_2(T)
P_A^\gamma
}
$$

For the simplest initial case:

$$
\alpha=1
$$

$$
\beta=1
$$

$$
\gamma=1.
$$

Therefore:

$$
\boxed{
r_1=k_1(T)P_AP_B
}
$$

and:

$$
\boxed{
r_2=k_2(T)P_A
}
$$

Both rates should be expressed per unit catalyst mass.

For example:

$$
r_1,r_2 \quad [{\rm mol\,g_{cat}^{-1}\,s^{-1}}].
$$

The units of $k_1$ and $k_2$ must be chosen accordingly.

For example, if pressure is expressed in bar:

$$
r_1=k_1P_AP_B
$$

requires:

$$
{\rm mol\,g_{cat}^{-1}\,s^{-1}\,bar^{-2}}
$$

while:

$$
r_2=k_2P_A
$$

requires:

$$
{\rm mol\,g_{cat}^{-1}\,s^{-1}\,bar^{-1}}.
$$

---

# 11. Temperature dependence of the rate constant

The reaction-rate constants depend on temperature according to the Arrhenius relationship.

The conventional form is:

$$
k=
A\exp
\left(
-\frac{E_a}{RT}
\right).
$$

For numerical work, it is convenient to define the rate constant at a reference temperature $T_{\mathrm{ref}}$:

$$
k(T_{\mathrm{ref}}).
$$

The rate constant at any temperature is then:

$$
k(T)=k(T_{\mathrm{ref}})\exp\left[-\frac{E_a}{R}\left(\frac{1}{T}-\frac{1}{T_{\mathrm{ref}}}\right)\right]
$$

For Reaction 1:

$$
k_1(T)=k_{1,\mathrm{ref}}\exp\left[-\frac{E_{a,1}}{R}\left(\frac{1}{T}-\frac{1}{T_{\mathrm{ref}}}\right)\right]
$$

For Reaction 2:

$$
k_2(T)=k_{2,\mathrm{ref}}\exp\left[-\frac{E_{a,2}}{R}\left(\frac{1}{T}-\frac{1}{T_{\mathrm{ref}}}\right)\right]
$$

The kinetic parameters are therefore initially:

$$
\boxed{
k_{1,\mathrm{ref}},
E_{a,1},
k_{2,\mathrm{ref}},
E_{a,2}
}
$$

with the reaction orders initially fixed.

---

# 12. Packed-bed differential equations

For Reaction 1:

$$
A+B\rightarrow C
$$

and Reaction 2:

$$
A\rightarrow D,
$$

the packed-bed reactor balances are:

### Reactant A

$$
\frac{dF_A}{dW}=-r_1-r_2
$$

### Reactant B

$$
\frac{dF_B}{dW}=-r_1
$$

### Desired product C

$$
\frac{dF_C}{dW}=+r_1
$$

### Undesired product D

$$
\frac{dF_D}{dW}=+r_2
$$

### Inert gas

$$
\frac{dF_I}{dW}=0
$$

These five coupled ordinary differential equations constitute the basic reactor model.

They are integrated from:

$$
W=0
$$

to:

$$
W=W_{\mathrm{cat}}.
$$

---

# 13. Calculation performed at each point in the reactor

For every evaluation of the differential equations:

### 1. Obtain the current molar flows

$$
F_A,F_B,F_C,F_D,F_I
$$

### 2. Calculate total molar flow

$$
F_T=\sum_iF_i
$$

### 3. Calculate mole fractions

$$
y_i=\frac{F_i}{F_T}
$$

### 4. Calculate partial pressures

$$
P_i=y_iP
$$

### 5. Calculate temperature-dependent rate constants

$$
k_1(T),\quad k_2(T)
$$

### 6. Calculate reaction rates

$$
r_1=k_1(T)P_AP_B
$$

$$
r_2=k_2(T)P_A
$$

### 7. Calculate the derivatives

$$
\frac{dF_A}{dW},
\frac{dF_B}{dW},
\frac{dF_C}{dW},
\frac{dF_D}{dW},
\frac{dF_I}{dW}
$$

### 8. Numerically integrate until

$$
W=W_{\mathrm{cat}}.
$$

---

# 14. Effect of catalyst mass

The catalyst mass explicitly appears through the integration range:

$$
0\leq W\leq W_{\mathrm{cat}}.
$$

For otherwise identical operating conditions:

$$
W_{\mathrm{cat}}\uparrow
$$

generally gives:

$$
X_A\uparrow.
$$

Thus the catalyst mass should be treated as one of the independent experimental variables.

A useful measure of the relationship between feed rate and catalyst amount is:

$$
\boxed{
\frac{F_{T,0}}{W_{\mathrm{cat}}}
}
$$

which has units such as:

$$
{\rm mol\,g_{cat}^{-1}\,s^{-1}}.
$$

Alternatively:

$$
\boxed{
\frac{W_{\mathrm{cat}}}{F_{T,0}}
}
$$

can be regarded as a catalyst-mass-normalized contact-time variable.

For heterogeneous catalytic experiments, quantities such as GHSV or WHSV are also commonly used, but the fundamental reactor equations can be solved directly using inlet flow and catalyst mass.

---

# 15. Temperature assumption

For the initial model, assume the fixed bed is isothermal:

$$
\boxed{
T(W)=T_{\mathrm{reactor}}
}
$$

and therefore:

$$
\frac{dT}{dW}=0.
$$

The temperature set point directly determines $k_1(T)$ and $k_2(T)$.

An energy balance is not required for the initial version.

---

# 16. Pressure assumption

For the initial model, assume the back-pressure regulator maintains a uniform reactor pressure:

$$
\boxed{
P(W)=P_{\mathrm{BPR}}
}
$$

and therefore:

$$
\frac{dP}{dW}=0.
$$

Pressure drop through the catalyst bed is initially neglected.

A pressure-drop model can be introduced later if required.

---

# 17. Outlet molar flow rates

After integration to:

$$
W=W_{\mathrm{cat}},
$$

the solution gives:

$$
\boxed{
F_{A,out},
F_{B,out},
F_{C,out},
F_{D,out},
F_{I,out}
}
$$

These quantities describe the complete reactor outlet.

The outlet total molar flow is:

$$
\boxed{
F_{T,out}=F_{A,out}+F_{B,out}+F_{C,out}+F_{D,out}+F_{I,out}
}
$$

Because the reactions can change the total number of gas molecules:

$$
F_{T,out}
$$

does not necessarily equal:

$$
F_{T,0}.
$$

---

# 18. Outlet composition

The outlet gas composition is calculated from:

$$
\boxed{
y_{i,out}=\frac{F_{i,out}}{F_{T,out}}
}
$$

Therefore:

$$
y_{A,out}=\frac{F_{A,out}}{F_{T,out}}
$$

$$
y_{B,out}=\frac{F_{B,out}}{F_{T,out}}
$$

$$
y_{C,out}=\frac{F_{C,out}}{F_{T,out}}
$$

$$
y_{D,out}=\frac{F_{D,out}}{F_{T,out}}
$$

$$
y_{I,out}=\frac{F_{I,out}}{F_{T,out}}.
$$

For the initial model, these mole fractions can be assumed to be the quantities reported directly by the simulated GC.

They must satisfy:

$$
\boxed{
y_{A,out}
+y_{B,out}
+y_{C,out}
+y_{D,out}
+y_{I,out}
=1
}
$$

---

# 19. Outlet volumetric flow rate

If desired, the actual outlet volumetric flow rate at reactor temperature and pressure can be calculated using the ideal-gas equation:

$$
\boxed{
\dot V_{out}=\frac{F_{T,out}RT}{P}
}
$$

This is the volumetric flow at the actual reactor $T$ and $P$.

If the outlet flow is to be reported as a standard volumetric flow, then:

$$
\boxed{
\dot V_{\mathrm{std},out}=\frac{F_{T,out}RT_{\mathrm{std}}}{P_{\mathrm{std}}}
}
$$

should instead be used.

It is important not to confuse actual reactor volumetric flow with standard MFC flow.

---

# 20. Conversion of A

The fractional conversion of reactant $A$ is:

$$
\boxed{
X_A=\frac{F_{A,0}-F_{A,out}}{F_{A,0}}
}
$$

The conversion should satisfy:

$$
0\leq X_A\leq1.
$$

---

# 21. Selectivity toward C

For the simple reaction network used here:

$$
A+B\rightarrow C
$$

and:

$$
A\rightarrow D,
$$

the selectivity toward $C$ may be defined as:

$$
\boxed{
S_C=\frac{F_{C,out}}{F_{C,out}+F_{D,out}}
}
$$

provided no products are present in the feed.

---

# 22. Yield of C

For the present stoichiometry, the yield of $C$ relative to $A$ is:

$$
\boxed{
Y_C=\frac{F_{C,out}}{F_{A,0}}
}
$$

and equivalently:

$$
\boxed{
Y_C=X_AS_C
}
$$

for this simple reaction network.

---

# 23. Inputs and outputs of the complete reactor model

The complete forward model can therefore be written as:

$$
\boxed{
\mathbf u=[F_{A,0},F_{B,0},F_{I,0},W_{\mathrm{cat}},T,P,\boldsymbol{\theta}]
}
$$

where:

$$
\boldsymbol{\theta}=[k_{1,\mathrm{ref}},E_{a,1},k_{2,\mathrm{ref}},E_{a,2}].
$$

The reactor model calculates:

$$
\boxed{
\mathbf y=[F_{A,out},F_{B,out},F_{C,out},F_{D,out},F_{I,out},F_{T,out}]
}
$$

and:

$$
\boxed{
[y_{A,out},
y_{B,out},
y_{C,out},
y_{D,out},
y_{I,out}]
}
$$

together with quantities such as:

$$
\boxed{
X_A,\quad S_C,\quad Y_C
}
$$

---

# 24. Overall calculation sequence

The complete calculation for one simulated experiment is:

```
MFC settings
    ↓
Convert standard volumetric flows to molar flows
    ↓
FA,0 , FB,0 , FI,0
    ↓
Specify catalyst mass Wcat
    ↓
Specify reactor temperature T
    ↓
Specify reactor pressure P
    ↓
Calculate Arrhenius rate constants
    ↓
Integrate packed-bed reactor ODEs
from W = 0 to W = Wcat
    ↓
Obtain outlet molar flows
    ↓
Calculate total outlet flow
    ↓
Calculate outlet mole fractions
    ↓
Calculate conversion, selectivity and yield
    ↓
Simulated GC result
```

The mathematical core of the entire initial digital reactor is therefore simply:

$$
\boxed{
\frac{d\mathbf F}{dW}=f(\mathbf F,T,P,\boldsymbol{\theta})
}
$$

with:

$$
\boxed{
\mathbf F(0)=[F_{A,0},F_{B,0},0,0,F_{I,0}]
}
$$

integrated until:

$$
\boxed{
W=W_{\mathrm{cat}}.
}
$$

This gives the gas composition and total flow leaving the catalytic fixed-bed reactor.
