@echo off

if "%1" NEQ "" set year=%1
if "%2" NEQ "" set num_files=%2
if "%3" NEQ "" set ok=%3

:: Check if arguments are provided, otherwise ask for input
:askYear
if "%year%"=="" set /P year="What year would you like to download? (last two digits, e.g. 12): "
if "%year%"=="" goto askYear

:askFiles
if "%num_files%"=="" set /P num_files="How many files are in this year's folder?: "
if "%num_files%"=="" goto askFiles

echo Download the first %num_files% files from the year 20%year%

:askOK
if "%ok%"=="" set /P ok="Is this input correct? (Y/N): "
if "%ok%"=="" goto askOK

if /I "%ok%" NEQ "Y" goto end

set /a i=1
set u=https://gml.noaa.gov/aftp/data/radiation/surfrad/Desert_Rock_NV/20
set u_=dra
set r=00
set l=.dat
set modr=10

mkdir surfrad\Desert_Rock_NV\20%year%

:start

set /a procent=%i% * 100 / %num_files%
TITLE Downloading %u%%year%\%u_%%year%%r%%i%%l% - [%procent%%%]
curl %u%%year%/%u_%%year%%r%%i%%l% -o surfrad\Desert_Rock_NV\20%year%\%u_%%year%%r%%i%%l%

set /a i+=1

if %i% == 10 set r=%r:~0,-1%
if %i% == 100 set r=%r:~0,-1%

if %i% LEQ %num_files% goto start

:end
