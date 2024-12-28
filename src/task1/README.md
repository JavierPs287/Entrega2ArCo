# Tarea 1
En esta tarea paralelizarás parte del código con DPC, en concreto deberás traducir a DPC la operación centroide (averagePixel).
Luego, en la siguiente tarea deberás realizar este mismo trabajo con las otras operaciones.

## Preparar el código
Al igual que con la práctica de DPC deberás descargar el fichero stimuli.bin y ponerlo en el mismo directorio que el ejecutable que generes tras realizar la compilación. El fichero se puede descargar, junto a los ficheros golden y los papers asociados al código desde el siguiente [enlace](https://mega.nz/folder/x4gRhLJJ#GRdxQc1Hnw3Lk_-9JC3Uew).

Recuerda que en esta práctica **no realizaremos la paralelización a nivel de bloque**, sino que paralelizaremos las diferentes operaciones convirtiéndolas en kernels (centroid, brightness, averagePixel...)

## Crea la cola y un selector de dispositivo por defecto
Busca cómo crear la cola y pasa como parámetro a la cola el selector por defecto. Recuerda que la cola se tiene que crear antes de cualquier tipo de computo. Es importante que entiendas que las colas se crean una única vez y por lo tanto se deberá ir compartiendo entre los diferentes métodos para que todos puedan mandar trabajos a ella. Crear una cola antes de cada operación tendría una sobrecarga muy alta y limitaría el rendimiento del programa final.

## ¿Para que sirve la cola?
La cola es el mecanismo de gestión de ejecución de tareas de forma paralela, ejecutando cada parte en un dispositivo específifco. Sus principales propósitos son enviar las tareas al dispositivo, especificar el dispositivo en el que queremos ejecutarlas, sincronizar las tareas, como en el caso en el que se usen recursos compartidos y se tenga que acceder a memoria en un orden determinado, y para optimizar el rendimiento mediante la paralelización de la ejecución

## Adapta los tipos de datos
Aunque este paso es opcional ya que DPC puede trabajar con tipos de datos de C, resulta interesante migrar las estructuras de datos básicas de C a C++. Por ejemplo, en lugar de usar un int [] en std::array<int, ...>. Trata de modificar paso a paso las estructuras que consideres más importantes y comprueba que el programa sigue funcionando. Hasta que no te hayas asegurado que el programa continua mostrando los resultados correctos no continues.

## Para cada array crea sus buffer correspondientes
Recuerda que uno de los modos de acceder a memoria en DPC es mediante los buffers y los accessors. Este es el método que vamos a usar en esta práctica. Para ello, crea los buffers una vez inicializados los espacios de memoria donde vamos a trabajar. Intenta mantener nombres coherentes para que te sea más sencillo identificarlos.

## Crea un rango para indicar el espacio de trabajo del kernel
El primer parámetro del paralell_for va a ser el rango de trabajo. Para definir este rango tenemos que indicar, en tiempo de compilación, el número de dimensiones que tiene y el tamaño de cada dimensión.


## Manda una tarea a la cola
Ahora crearemos el trabajo dentro de la cola donde, en las siguientes tareas, iremos metiendo el código para: solicitar acceso a las variables y ejecutar el trabajo. Busca como crear el trabajo para la cola. Comprueba que el código compila.

## Solicitar acceso a los buffer
Recuerda que hemos creado los buffer para abstraer el acceso a las diferentes variables. Tal y como se ha visto en teoría hay que solicitar el acceso a cada buffer por medio de los accessors. **En las primeras líneas dentro de la cola** define los accessor que necesitamos.

Pistas:
* Elige bien los permisos.

## Crea el parallel_for
Ahora que ya tenemos acceso a cada uno de los buffer, hay que solicitar por medio del handler el parallel_for indicando: el rango del kernel (revisa los pasos anteriores) y el kernel. Inicialmente deja el kernel vacio y comprueba que la compilación es correcta.

Una vez finalizado el parallel_for (recuerda añadir el código del cálculo del centroide en el kernel), compila el código y asegurate de no tener errores de compilación.

Pistas:
* Busca en el código de referencia. Lo más probable es que debáis adaptar el código de referencia a la estructura seguida por DPC.

## Esperar al trabajo y acceder a los resultados
Finalmente solo queda esperar a la cola, busca como puedes esperar y acceder mediante un host_accessor al buffer de salida del kernel. Únicamente creando el host_accessor al buffer el runtime se asegura que tienes acceso al mismo y ya puedes acceder como lo harías normalmente.

**Comprueba que el código produce los mismos resultados que el código secuencial**
**Guarda el código final en results/task1/**

----

# Task 1
In this task, you will parallelize part of the code using DPC. Specifically, you will need to translate the centroid operation to DPC.

## Prepare the code
First, locate the code corresponding to the centroid and how the hyperspectral cube is read from the image hosted on [mega](https://mega.nz/file/Z5JUkSoI#boptGx0TD4YU1FGz5WxVkxgB0-fav1sQiVVCk2lz_CA). Remember that this time we will only use one block, so you need to modify the amount of data to be read.

## Create the queue using a default device selector
Find out how to create the queue and pass the default device selector as a parameter to the queue. Remember that the queue has to be created before any computation.


## What is the queue used for?
**Answer here**

## Create the corresponding buffer for each array
Remember that one of the ways to access memory in DPC is through buffers and accessors. This is the method we will use in this practice. To do this, create the buffers once the memory spaces where we will work are initialized, i.e., just before the for loop.


## Create a range for the kernel
The first parameter of the *parallel_for* will be the workspace range. To define this range, we have to indicate, at compile time, the number of dimensions it has and the size of each dimension.


## Submit a task to the queue
Now we will create the job inside the queue where, in the following tasks, we will insert the code to: request access to variables and execute the job. Find out how to create the job for the queue. Verify that the code compiles.


## Request access to buffers
Remember that we have created buffers to abstract access to different variables. As seen in theory, you have to request access to each buffer through accessors. **In the first lines within the queue**, define the accessors that we need.

Clues:
* Choose the right permissions.

## Create the parallel_for
Now that we have access to each buffer, we need to request, through the handler, the parallel_for indicating: the kernel's range (review the previous steps) and the kernel itself. Initially, leave the kernel empty and check that the compilation is correct.

Once the parallel_for is completed (remember to add the centroid calculation code in the kernel), compile the code and make sure there are no compilation errors.

Clues:
* Look in the reference code. Most likely, you will need to adapt the reference code to the structure followed by DPC.

## Wait for the task and get the results
Finally, all that's left is to wait for the queue. Find out how to wait and access the kernel's output buffer using a host_accessor. By creating the host_accessor to the buffer, the runtime ensures that you have access to it, and you can access it as you normally would.


**Verify that the code produces the same results as the sequential code**
**Save the final code in results/task1/dpcHLCA.cpp**
