<h4 align="center"> Towards a framework for incorporating data acquisition cost in predictive time series models </h4>

<h5 align="center"> 6TH WORKSHOP ON MINING AND LEARNING FROM TIME SERIES (MiLeTS) </h5> 

<h5 align="center"> KDD 2020</h5>

<h4 align="center"> A. Stetco, R. Mosincat, G. Nenadic, J. Keane </h4> 

<p align="center">
  <a href="https://kdd-milets.github.io/milets2020/papers/MiLeTS2020_paper_12.pdf">Paper</a></br>
  <a href="https://drive.google.com/file/d/1R4pvN71zCRz3sd4MZQmPYmY6fESLbirM/view">Presentation</a></br>
  <a href="https://drive.google.com/file/d/1UMZcCpHieuHXYLDvP-uyMNpfpoa2OHNC/view">Slides</a></br>
</p>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In many real world applications of machine learning, the cost of acquiring measurements varies significantly. As businesses operate in resource-constrained environments, a question that arises is which combination of signals results in the most accurate models given a fixed budget. Conversely, if more accurate models are needed, what signals increase accuracy at lowest cost? 

This code introduces a three-stage framework for integrating data acquisition cost into time-series applications based on signal ablation. Stage 1 constructs a time-series predictive model utilizing all relevant signals to assess the maximum accuracy achievable with an unlimited budget. Stage 2 randomizes each signal independently, computing the degradation in accuracy incurred on the underlying model. This process is achieved either by shuffling of its measurements or random sampling from signal’s distribution. Utilizing the resulting estimated signals accuracy and their acquisition cost, Stage 3 solves the knapsacklike combinatorial optimization problem of picking the right combination of signals maximizing total accuracy given any fixed budget. 

<p align="center">
  <img alt="Framework" src="https://github.com/adrianstetco/CostFrameworkML/blob/master/framework.png" width=400"/> 
  <h5  align="center"> Figure 1. Framework for Cost Optimization</h5>                                                                                                           
</p>

The proposed framework is showcased on a synthetic time-series dataset which allows us to control the cost distribution in order to understand how the system works in different scenarios.

<p align="center">
  <div>
    <img alt="Framework" src="https://github.com/adrianstetco/CostFrameworkML/blob/master/signals.png" width=380">
    <h5> Figure 2. Framework for Cost Optimization</h5>                                                                                                           
  </div>                                                                                                                 
  <div>                                                                                                                 
    <img alt="Framework" src="https://github.com/adrianstetco/CostFrameworkML/blob/master/target.png" width=300">
    <h5> Figure 3. Framework for Cost Optimization</h5>     
  </div>
</p>
  

 
