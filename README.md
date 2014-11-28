ccminer
=======

Based on Christian Buchner's & Christian H.'s CUDA project

Forked from tpruvot@github

Fork by sp-hash@github with X14,X15,X17,WHIRL and Blake256 support (NEOS + BlakeCoin), and some others, check the [README.txt](README.txt)

   BTC donation address: 
	sp-hash@github:
		BTC: 1CTiNJyoUmbdMRACtteRWXhGqtSETYd6Vd
		DRK: XdgfWywdxABwMdrGUd2xseb6CYy1UKi9jX

A part of the recent algos were originally written by [djm34](https://github.com/djm34).

This variant was tested and built on Linux (ubuntu server 14.04) and VStudio 2013 on Windows 7.

Note that the x86 releases are generally faster than x64 ones on Windows.

About source code dependencies
------------------------------

This project requires some libraries to be built :

- OpenSSL (prebuilt for win)

- Curl (prebuilt for win)

- pthreads (prebuilt for win)

The tree now contains recent prebuilt openssl and curl .lib for both x86 and x64 platforms (windows).

To rebuild them, you need to clone this repository and its submodules :
    git clone https://github.com/peters/curl-for-windows.git compat/curl-for-windows

There is also a [Tutorial for windows](http://cudamining.co.uk/url/tutorials/id/3) on [CudaMining](http://cudamining.co.uk) website.

