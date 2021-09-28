---
title: 
feature_text: |
  ## Green AI UPPA 
feature_image: "/assets/ossau_lurien.png"
excerpt: "Longer intro"
aside: false 
---

## Introduction

The Green AI research team develops solutions for people working with data or intersted in data and environment. 

Based on the two articles written by Sébastien Loustau ([1](https://hal.archives-ouvertes.fr/hal-03262679v4/document), [2](https://hal.archives-ouvertes.fr/hal-03262687v2/document)), we want to extend these results using error bounds to create new algorithms. These algorithms must be a part of the answer to ecological problem like global warming.
Indeed, the Machine Learning and deep learning are more and more popular and easily approachable with new applications every day.
Machine Learning and Deep Learning include the phase of training and inference/prediction. These are really especially consuming in term of energy. Notice that in 2019, the electrical consumption of IT represent more than 3% of the worldwide consumption and increases by 6% per year (The Shift Project 2020, 2021).

For example, a classical model ALexNet has 60 millions of parameters. To realize the measure of consumption, some libraires are already available as *psutil*... We also are implementing our own libraires, find them on [git](https://github.com/GreenAI-Uppa/deep_learning_power_measure).


Many approachs can be done here, for example quantization, pruning or binarization, act directly on the computation of weights by reducing their numbers, by changing their values or by selecting features of interest (lasso).

Using this kind of technics and theorical results, we aim to write new algorithms. You probably already know some light model like bertlight or mobilenet but their main purpose is the efficiency and the bias-variance tradeoff, our point of view is accuracy-energic consumption ratio.

Today we are collaborating with our first industrial partners about projet including climate.
Several prototypes are in progress.
**[Prof en Poche](#ai-for-education)** allow us to apply our new model on education data. 
With **[Altanoveo](#crisis-management)**, we are able to developpe climate crisis model. 
FInally, we work with **Mallisphera** to understand farming practices and their CO² emission.

For more details, find articles, codes sources, references, formations and contact below. 

## Our research

![seminar4](/images/seminar4.png){:class="img-responsive"}

Our research encompasses diverse projects and collaborations around the mathematical foundations of power-efficient deep/machine learning algorithms, and the applications of AI to build a more sustainable world. In this section, we present the mathematical as well as the algorithmic part of our activity. 

### Our algorithms 

Deep Learning has become extremely popular to solve many supervised machine learning problems. This standardization of machine learning, namely computing on GPU a stochastic gradient descent is not only a plague for science but also a disaster in terms of power consumption. Recently, a growing interest is observed in the deep learning hype in order to reduce computational cost by designing lighter architectures. Several approaches for reducing the computational effort of NNs have been proposed (e.g. binarized networks or pruning methods). Moreover, promising strategies propose to select the connectivity of the network, or more generally the architecture, during the training process. 

 GreenAI Uppa expect to address these issues, based on both a theoretical and practical machine/deep learning analysis of standard pipeline and new paradigms. More precisely, we propose alternatives to standard deep learning pipeline in order to rethink the learning process and show how deep mathematical statistics could help us to select lighter algorithms and reduce training, inference complexity and environmental impact of machine learning.

### Measure the hungriness of your deep learning

We are also mainly focus on more a practical study of recent architectures in terms of power consumption, by using recent softwares like RAPL and nvidia-smi in a dedicated software available here. Gathering with a mathematical background, we have the long term objective to design adaptive architectura that fits dynamically to the size of the data, the difficulty of the problem and the desire environmental constraints.

### Seminars and upcoming events

- **Upcoming WORKSHOP at ACML 2021, November 17th** Power-Efficient Deep Learning. Organized with [Pierre Alquier](https://pierrealquier.github.io/) from the amazing [Approximate Bayesian Inference Team](https://team-approx-bayes.github.io/) at [Riken Institute](https://www.riken.jp/en/). Details in a dedicated landing page asap.
- **Next team seminar on 11th, October** Matthieu François
- **25th, October** Paul Gay
- **8th, November** Jordy Palafox
- **22th, November** Sébastien Loustau
- **6th, December** Mellisphera project
[expand]

 - **September, 27th 2021 - Sébastien Loustau (UPPA)** Introduction to convex optimization
Summary : In this lecture, I will introduce convex optimization theory and mirror descent. We start with a theoretical motivation and studyt of (stochastic) gradient descent, and then moove to the non-euclidean setting to derive mirror descent algorithm as a generalization of gradient descent. We finally apply it to the context of expert advices to recover the classical regret bound for exponential weighted averages previously presented in the first seminar in july, and discuss possible applications to Green AI.
<br>[slides here]('/seminars/sem5.pdf')
 - **September, 13th 2021 - Sébastien Loustau (UPPA)** Kick Off Green AI Uppa
Summary : Official kick off of the Green AI Uppa project ! After presenting the climatic and mathematical motivations (has the earth ever been this hot before ?), we introduce the context and support we have from the public institutions and the SMEs. We explain how the team will be organized, and inspired from the best of both worlds. Then we take 30 minuts to write our **elevator pitch**. Welcome to Jordy and Matthieu !
<br>[slides here]('/seminars/sem4.pdf')
 - **August, 23th 2021 - Paul Gay (UPPA)** Measuring the Power draw of computers
Summary : The ability of measuring power and consumption of machine learning algorithms is necessary to design new efficient ones. Nowadays, there is a growing interest in the machine learning and IT community for measuring the consumption at different scale, from the AI model to the entire data center. In this talk, we survey recent tools and softwares based on RAPL and NVIDIA-SMI and highlight the dependancy to the hardware considered (CPU, GPU), as well as the different sources of consumption in the architecture of a computer. The final goal is to give to engineers and data scientists the capacity to measure the consumption of deep/machine learning algorithms via our open source software [deep_learning_power_measure](https://github.com/GreenAI-Uppa/deep_learning_power_measure) developed by Green AI Uppa.
<br>[slides here]('/seminars/sem3.pdf')
 - **July, 26th 2021 - Julien Mercier (UPPA)** How to binarize a neural network ?
Summary : In this talk, I propose to present the main pros and cons of binarization via the gradient. We present two main historical attempt: [BinaryConnect](https://arxiv.org/abs/1511.00363) and [BinaryNetwork](https://arxiv.org/abs/1602.02830), based on two recent papers.
<br>[slides here]('/seminars/sem2.pdf')
 - **July, 5th 2021 - Sébastien Loustau** Comment intégrer des contraintes environnementales dans les algorithmes de Deep Learning ?
Résumé : Dans cet exposé, je vous propose d'introduire la théorie PAC-Bayesienne et son lien avec la théorie des jeux pour proposer de nouvelles procédures d'apprentissage séquentiel. Après une introduction simple du problème de prévision avec avis d'experts, on présentera des inégalités PAC-Bayesiennes plus générales et des bornes de regrets parcimonieuses pour des famillles de réseaux de neurones particulières (XNOR-nets notamment). Enfin, on proposera une extension de cette théorie au Transport Optimal pour intégrer de nouvelles pénalités plus génériques et garantir l'intégration de mesures de consommations plus précises.
Cet exposé est une introduction aux principales motivations de ces articles : Learning with BOT et Sparsity regret bounds for XNOR-Nets
notes manuscrites disponibles ici
<br>[slides here]('/seminars/sem1.pdf')
[/expand]

### Training

We have a long experience in creating training content on machine learning and deep learning towards student and industrial learners 

Contact us if you are interested. 
## Industrial projects

### Crisis management

ALTANOVEO is specialised in very high resolution visualization and has developped for 10 years CWall. 

CWall is a high resolution collaborative software solution designed to be deployed on video walls.

CWall is multi-site, multi-source and multi-application. It offers real-time collaboration among people working in the same room or remotely by sharing thier computer screens, video streams,  web application and so on. 

CWall shows its full potential for : 

joint study of resources at a very high resolution across a wide range of disciplines, 
simultaneous display of several streams and information into different windows, particularly in crisis 
management or monitoring situations.

![example](images/ALTANOVEO-4660-1024x683.jpg "Title")

Our main purpose is to integrate AI on Cwall. This process will help to do crisis management with KPIs, advices... 

Based on low consumption algorithms, this project start with an analysis of social networks and open source data as climate crisis images. Being able to summarize open informations and foresee their impact will be a great addition upgrade to this solution. 

### AI for Education


The start-up [Proche en Poche](https://profenpoche.com/) was founded in 2015 by Vincent Escudé and Samuel Imbert. 
It is specialized in creation on new numerical educative solutions and more specifically, propose to provide a teacher to a needed student using smartphone, tablet or computer. They also created Mathia using AI as a innovative product in order to give them the love of mathematics of primary school children which is based on AI. In 2019, they were rewarded for Mathia as laureate of P2IA (Partenariat d’Innovation en Intelligence Artificielle) launched by the Ministère de l'Education Nationale et de la Jeunesse.

Mathia is an app with a recommandation system of studies for children, based on a cluster of children considering to their own capacities and difficulties.
Our goal is to improve these tasks and reducing the consumption of energies.


## Sponsors

This project is mainly supported by the I-Site E2S-UPPA, a french consortium composed of [CNRS](https://www.cnrs.fr/), [INRIA](https://www.inria.fr/), [INRAE](https://www.inrae.fr/) that assists companies and public organizations in their energy and environmental transition.

Partnerships with public institutions, as well as SMEs are one of the main motivations for the existence of GreenAI Uppa. 
 - [AltaNoveo](https://www.cwall.fr/)
 - [Aquitaine Science Transfert](https://www.ast-innovations.com/)
 - [Technopole Hélioparc](https://helioparc.fr/)
 - [Mellisphera](https://www.mellisphera.com/)
 - [ORIE, Cornell University](https://www.orie.cornell.edu/orie)
 - [Prof en Poches](https://profenpoche.com/)
 - [Réseau Francilien en Sciences Informatiques (RFSI)](https://dim-rfsi.fr/)


## Who are we ?


### Sébastien Loustau

After a PHD thesis defended in 2008, at [Institut de Marseille](https://www.i2m.univ-amu.fr) he worked as Associated professor at [Université d'Angers](https://www.univ-angers.fr/fr/index.html) in the lab [LAREMA](https://math.univ-angers.fr/LAREMA/) until 2015 and he defended his Habilitation thesis in 2014.
He found the AI start-up [LumeanAI](https://www.lumenai.fr/) in 2015 and the non-profit organization [IAPau](http://www.iapau.fr/) in 2016.
Since 2020, he is researcher at the [LMAP](https://lma-umr5142.univ-pau.fr/fr/index.html) at [UPPA](https://www.univ-pau.fr/fr/index.html) and found with Paul Gay the Green IA Uppa Group.
His reshearch interest are online learning, mathematical statistics, information theory for machine learning  and more recently, in the applications of these activities to deep learning techniques and environmental challenges. 
For contact, send an email at sebastien[dot]loustau[at]univ-pau.fr

[Web page](https://sebastienloustau.github.io/)

### Paul Gay

Currently reshearcher engineer at the [LMAP](https://lma-umr5142.univ-pau.fr/fr/index.html), he graduated in computer sciences from [INSA Rouen](http://asi.insa-rouen.fr/) from 2006 to 2011. He did his PHD between [LIUM](https://lium.univ-lemans.fr/) in France and  [IDIAP Reshearch Institut](https://www.idiap.ch/en) in Switzerland up to 2014  on unsupervised Audio-visual person identification in broadcast data. After a period of Teaching in Avignon in ATER position in [LIA](https://lia.univ-avignon.fr/) and a Post-doc in Italy, at [IIT/PAVIS](https://pavis.iit.it/) (2014 - 2018), he joined LumenAI as R&D Engineer from 2019 to 2020.
For contact, send an email at paul[dot]gay[at]univ-pau.fr 

[Web page](https://paulgay.github.io/index.html)

### Matthieu Francois

He obtained his Master's degree in 2020 at [Université de Grenoble Alpes](https://www.univ-grenoble-alpes.fr/). After some experiences as Data Consultant, he joined Green AI UPPa Team in Septembre 2021.
For contact, send an email at matthieu[dot]francois[at]univ-pau.fr \
[Web page](https://www.linkedin.com/in/matthieufran%C3%A7ois/)

### Jordy Palafox

He defended his PHd Thesis in 2018 at [UPPA](https://www.univ-pau.fr/fr/index.html) [LMAP](https://lma-umr5142.univ-pau.fr/fr/index.html). After a post-doc in 2018-2019 at La Rochelle in the lab [LASIE](https://lasie.univ-larochelle.fr/), he taught at CY Tech two years and followed training in Data Science. He joined Green AI Pau group in Septembre 2021.
For contact, send an email at palafox[dot]jordy[at]univ-pau.fr \
[Web page](https://www.linkedin.com/in/jpalafox1242/)

## Contact

Feel free to contact us if you want to contribute: contact [Paul](https://paulgay.github.io/) or [Sébastien](https://sebastienloustau.github.io)



## Start the different parts 

- Available as a **theme gem** and **GitHub Pages** theme
- Clear and elegant design that can be used out of the box or as solid starting point
- Tested in all major browsers, including **IE and Edge**
- Built in **Service Worker** so it can work offline and on slow connections
- **Configurable colours** and typography in a single settings file
- Extensive set of **shortcodes** to include various elements; such as buttons, icons, figure images and more
- Solid **typographic framework** from [Sassline](https://sassline.com/)
- Configurable navigation via a single file
- Modular Jekyll components
- Post category support in the form of a single post index page grouped by category
- Built in live search using JavaScript
- **Contact form** built in using [Formspree](https://formspree.io/)
- Designed with **[Siteleaf](https://www.siteleaf.com/)** in mind
- Has 9 of the most popular networks as performant sharing buttons
- Has documentation

## link

## Examples

Here are a few examples of Alembic out in the wild being used in a variety of ways:

- [bawejakunal.github.io](https://bawejakunal.github.io/)
- [case2111.github.io](https://case2111.github.io/)
- [karateca.org](https://www.karateca.org/)

## Installation

### Quick setup

To give you a running start I've put together some starter kits that you can download, fork or even deploy immediately:

- ⚗️🍨 Vanilla Jekyll starter kit  
  [![Deploy to Netlify](https://www.netlify.com/img/deploy/button.svg)](https://app.netlify.com/start/deploy?repository=https://github.com/daviddarnes/alembic-kit){:style="background: none"}
- ⚗️🌲 Forestry starter kit  
  [![Deploy to Forestry](https://assets.forestry.io/import-to-forestry.svg)](https://app.forestry.io/quick-start?repo=daviddarnes/alembic-forestry-kit&engine=jekyll){:style="background: none"}  
  [![Deploy to Netlify](https://www.netlify.com/img/deploy/button.svg)](https://app.netlify.com/start/deploy?repository=https://github.com/daviddarnes/alembic-forestry-kit){:style="background: none"}
- ⚗️💠 Netlify CMS starter kit  
  [![Deploy to Netlify](https://www.netlify.com/img/deploy/button.svg)](https://app.netlify.com/start/deploy?repository=https://github.com/daviddarnes/alembic-netlifycms-kit&stack=cms){:style="background: none"}

- ⚗️:octocat: GitHub Pages with remote theme kit  
  {% include button.html text="Download kit" link="https://github.com/daviddarnes/alembic-kit/archive/remote-theme.zip" color="#24292e" %}
- ⚗️🚀 Stackbit starter kit  
  [![Create with Stackbit](https://assets.stackbit.com/badge/create-with-stackbit.svg)](https://app.stackbit.com/create?theme=https://github.com/daviddarnes/alembic-stackbit-kit){:style="background: none"}

### As a Jekyll theme

1. Add `gem "alembic-jekyll-theme"` to your `Gemfile` to add the theme as a dependancy
2. Run the command `bundle install` in the root of project to install the theme and its dependancies
3. Add `theme: alembic-jekyll-theme` to your `_config.yml` file to set the site theme
4. Run `bundle exec jekyll serve` to build and serve your site
5. Done! Use the [configuration](#configuration) documentation and the example [`_config.yml`](https://github.com/daviddarnes/alembic/blob/master/_config.yml) file to set things like the navigation, contact form and social sharing buttons

### As a GitHub Pages remote theme

1. Add `gem "jekyll-remote-theme"` to your `Gemfile` to add the theme as a dependancy
2. Run the command `bundle install` in the root of project to install the jekyll remote theme gem as a dependancy
3. Add `jekyll-remote-theme` to the list of `plugins` in your `_config.yml` file
4. Add `remote_theme: daviddarnes/alembic@main` to your `_config.yml` file to set the site theme
5. Run `bundle exec jekyll serve` to build and serve your site
6. Done! Use the [configuration](#configuration) documentation and the example [`_config.yml`](https://github.com/daviddarnes/alembic/blob/master/_config.yml) file to set things like the navigation, contact form and social sharing buttons

### As a Boilerplate / Fork

_(deprecated, not recommended)_

1. [Fork the repo](https://github.com/daviddarnes/alembic#fork-destination-box)
2. Replace the `Gemfile` with one stating all the gems used in your project
3. Delete the following unnecessary files/folders: `.github`, `LICENSE`, `screenshot.png`, `CNAME` and `alembic-jekyll-theme.gemspec`
4. Run the command `bundle install` in the root of project to install the jekyll remote theme gem as a dependancy
5. Run `bundle exec jekyll serve` to build and serve your site
6. Done! Use the [configuration](#configuration) documentation and the example [`_config.yml`](https://github.com/daviddarnes/alembic/blob/master/_config.yml) file to set things like the navigation, contact form and social sharing buttons

## Customising

When using Alembic as a theme means you can take advantage of the file overriding method. This allows you to overwrite any file in this theme with your own custom file, by matching the file name and path. The most common example of this would be if you want to add your own styles or change the core style settings.

To add your own styles copy the [`styles.scss`](https://github.com/daviddarnes/alembic/blob/master/assets/styles.scss) into your own project with the same file path (`assets/styles.scss`). From there you can add your own styles, you can even optionally ignore the theme styles by removing the `@import "alembic";` line.

If you're looking to set your own colours and fonts you can overwrite them by matching the variable names from the [`_settings.scss`](https://github.com/daviddarnes/alembic/blob/master/_sass/_settings.scss) file in your own `styles.scss`, make sure to state them before the `@import "alembic";` line so they take effect. The settings are a mixture of custom variables and settings from [Sassline](https://medium.com/@jakegiltsoff/sassline-v2-0-e424b2881e7e) - follow the link to find out how to configure the typographic settings.
