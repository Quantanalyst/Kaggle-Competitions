## Computing Tools and Environment


## Colab
One good way to use colab is to connect it with GitHub. 

Note1: When there is a _large dataset_, use **Git LFS**. Some necessary instruction for future reference:
  * Install Git Large File System (LFS):
    * Navigate to git-lfs.github.com and click Download.
    * After download, go to the directory and run the below command
      * $ sudo ./install.sh
    * Verify that the install was successful
      * $ git lfs install
  * After install git lfs, then you need to determine which files must be stored with lfs. You can assign those files with assigning their extension.
  * set up Git LFS and its respective hooks by running:
    * $ git lfs install  (You'll need to run this in your repository directory, once per repository.) 
  * Select the file types you'd like Git LFS to manage (or directly edit your .gitattributes). You can configure additional file extensions at anytime.
    * $ git lfs track "*.psd"
  * Make sure .gitattributes is tracked
    * $ git add .gitattributes
