# Characterizing the performance of the SPHERE exoplanet imager at the Very Large Telescope using deep learning
<a name="readme-top"></a>

<!-- TABLE OF CONTENTS -->
<details>
<summary>Table of Contents</summary>
    <ol>
        <li>
            <a href='#About-the-project'>About the project</a>
        </li>
        <li>
        <a href="#Database-creation">Database creation</a>
            <ul>
                <li><a href="#Prerequisites">Prerequisites</a>
            </ul>
            <ul>
                <li><a href="#SPHERE-DATA">SPHERE DATA</a>
            </ul>
        </li>
        <li>
            <a href='#Contact'>Contact</a>
        </li>
    </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About the project

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->

Taking direct pictures of extrasolar planetary systems is an important, yet challenging goal of modern astronomy, which requires specialized instrumentation. The high-contrast imaging instrument SPHERE, installed since 2014 at the Very Large Telescope, has been collecting a wealth of data over the last eight years. An important aspect for the exploitation of the large SPHERE data base, the scheduling of future observations, and for the preparation of new instruments, is to understand how instrumental performance depends on environmental parameters such as the strength of atmospheric turbulence, the wind velocity, the duration of the observation, the pointing direction, etc. With this project, we propose to use deep learning techniques in order to study how these parameters drive the instrumental performance, in an approach similar to the one used by Xuan et al. 2018. This project will make use of first-hand access the large SPHERE data base through the SPHERE Data Center at IPAG/LAM (Grenoble/Marseille).

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Database creation

To get a local copy up and running follow these simple steps.

### Prerequisites

If you are using windows you should install [wget](https://gnuwin32.sourceforge.net/packages/wget.htm) and add its path to your environment variables.

### SPHERE data

You can either retrieve the SPHERE data download script from the [SPHERE client](https://sphere.osug.fr/spip.php?rubrique34&lang=fr) or simply use the one located in `DB_creation/Download Scripts`. 
Note that if you download the script yourself you need to parse it using `Dataset_creation/sphere_dl_parser.py` in order to skip unwanted (and voluminous) files.

Then you just have to launch a terminal in the script folder and execute it.

* Linux
  ```sh
  ./parsed_sphere_dl_script_contrast_curves.sh
  ```

* Windows
  ```sh
  sh parsed_sphere_dl_script_contrast_curves.sh
  ```

The observations will then be downloaded in the folder `SPHERE DC DATA` (located in the same directory as `parsed_sphere_dl_script_contrast_curves.sh`).

<!-- ### Installation

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/github_username/repo_name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ``` -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Ludo Bissot - ludo.bissot@student.uliege.be

Project Link: [https://github.com/lbissot/Master-Thesis](https://github.com/lbissot/Master-Thesis)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">back to top</a>)</p>