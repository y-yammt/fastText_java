# fasttext_java
Java port of [c++ facebook fastText](https://github.com/facebookresearch/fastText) 
and fork of [ivanhk fastText_java port](https://github.com/ivanhk/fastText_java).

The main differences with the original port:
* synchronization with c++ version
* support any kind of I/O streams (including hadoop and web in extra)
* __tests__
* java8
* changes in java-coding & OOP styles

# Version
### release c++ version: [0.1.0](https://github.com/facebookresearch/fastText/releases/tag/v0.1.0)
### last checked c++ revision number: [09/12/2017](https://github.com/facebookresearch/fastText/commit/b928c9f01d02fcf2f115f06ee7a2c02d5c6a0ca2)
### bin model version: 12

### Maven
    <dependency>
        <groupId>com.github.sszuev</groupId>
        <artifactId>fasttext</artifactId>
        <version>1.0.0</version>
    </dependency>
### Tools
to build command-line tools use `mvn package -Pmain` or `mvn package -Pextra`



