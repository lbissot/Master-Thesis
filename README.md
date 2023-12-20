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
            <a href='#Models'>Models</a>
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
## Dataset creation

To get a local copy up and running follow these simple steps.

### Prerequisites

If you are using windows you should install [wget](https://gnuwin32.sourceforge.net/packages/wget.htm) and add its path to your environment variables.

### SPHERE data

1. **Download SPHERE Data:**
    - You can either retrieve the SPHERE data download script from the [SPHERE client](https://sphere.osug.fr/spip.php?rubrique34&lang=fr) or simply use the one located in `Dataset_creation/Download Scripts`.  

    In order to retrieve the data from the [SPHERE client](https://sphere.osug.fr/spip.php?rubrique34&lang=fr) you need to follow these steps :

    i. **Browse Process**

    ii. **Use the following options :**
    ```plaintext
    Process/recipe/ird_specal_dc
    ```

    ```plaintext
    Process/Preset/cADI_softsorting / ird_specal_dc / production
    ```

    ```plaintext
    Observation / Parameters/Filter/DB_H23
    ```

    iii. **Generate the download script :**  
    Select all the processes, then right click and select `Download script/Selected - outputs only` 

    iv. **Parse the file :**
    In the `Dataset_creation` folder there is a file named `sphere_dl_parser.py` that can be used in order to remove the unwanted files in the download script.

    v. **Execute the script :**
    You just have to launch a terminal in the script folder and execute it.

   * Linux
    ```sh
    ./parsed_sphere_dl_script_contrast_curves.sh
    ```

    * Windows
    ```sh
    sh parsed_sphere_dl_script_contrast_curves.sh
    ```

2. **Create Data Folders:**
   - After downloading the data, create the following folder structure:

     ```plaintext
     SPHERE_DC_DATA
     ├── contrast_curves
     └── timestamps
     ```

     These folders are where the (raw) data from the SPHERE client should be placed.

3. **Move Data to Dataset Creation Folder:**
   - The observations will be downloaded in the folder `SPHERE_DC_DATA` (located in the same directory as `parsed_sphere_dl_script_contrast_curves.sh`).
   - Move the `SPHERE_DC_DATA` folder inside the `Dataset_creation` folder.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Models

In order to be able to track the training of the models as well as to hypertune them [Weights and Biases](https://wandb.ai) was used. Thus in order to be able to run the codes, you need to provide your login key in the `main` of the different files.

* `rf_single_nsigma.py` : Random forest that predicts a contrast value at a given separation.

* `nn_single_nsigma.py` : Neural Network that predicts a contrast value at a given separation.

* `nn_vector_nsigma.py` : Neural Network that predicts the whole contrast vector at once.

* `nn_single_uncertainty.py` : Neural Network that computes the aleatoric uncertainty. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Ludo Bissot - ludo.bissot@uclouvain.be

Project Link: [https://github.com/lbissot/Master-Thesis](https://github.com/lbissot/Master-Thesis)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Julien Milli's github](https://github.com/jmilou/sparta) has been used to query Simbad.

<p align="right">(<a href="#readme-top">back to top</a>)</p>